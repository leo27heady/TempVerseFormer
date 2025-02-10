import random

import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import IterableDataset

from .simple_shape_rotation import Scene, SceneType, Random2DShapeCreator, ShapeRotator, CreateShapeException
from ..config import DataConfig, TemporalPatterns


class ShapeDataset(IterableDataset):
    def __init__(self, device, config: DataConfig):
        self.scale_factor = 1000
        self.device = device
        self.config = config
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def scale(self, var: int | float) -> int:
        return int(var * self.scale_factor)

    def __iter__(self):
        triangle = Random2DShapeCreator().create_equilateral_triangle()
        
        scene = Scene(
            triangle, SceneType.DIM_2, self.config.render_window_size,
            bg_color="white", 
            mesh_color="black",
            show_edges=False,
            lighting=False,
            line_width=4.0,
            fixed_camera_distance=2.7, 
            axis="z"
        )
        
        while True:
            current_batch_time = random.randint(self.config.time_to_pred.min, self.config.time_to_pred.max)
            
            batch_images,batch_angles, batch_temp_patterns = [], [], []
            for batch in range(self.config.batch_size):
                
                # Selected imperially to fit the triangle to the render window, should be adjusted carefully
                base = random.randint(self.scale(0.5), self.scale(1)) / self.scale_factor
                shift = random.randint(self.scale(-0.2), self.scale(base + 0.2)) / self.scale_factor
                height = random.randint(self.scale(0.5), self.scale(1)) / self.scale_factor
                figure = Random2DShapeCreator().create_triangle(base, shift, height)
                
                # Prepare scene with the new figure, memory efficient - the scene remains the same
                scene.prepare_scene(
                    figure,
                    bg_color="white", 
                    mesh_color="black",
                    show_edges=False,
                    lighting=False,
                    line_width=4.0,
                    distance_factor=1.0,
                    fixed_camera_distance=2.7, 
                    axis="z"
                )
                figure.rotate_z(random.randint(0, self.scale(360)) / self.scale_factor, point=scene.center_of_mass, inplace=True)
                scene.plotter.render()

                angle = random.randint(self.scale(self.config.angle.min), self.scale(self.config.angle.max)) / self.scale_factor
                step = random.choice((-angle, angle))

                selected_temporal_patterns = self.config.temporal_patterns.copy()
                if TemporalPatterns.ACCELERATION in self.config.temporal_patterns and TemporalPatterns.DECELERATION in self.config.temporal_patterns:
                    if random.choice((True, False)):
                        selected_temporal_patterns.remove(TemporalPatterns.ACCELERATION)
                    else:
                        selected_temporal_patterns.remove(TemporalPatterns.DECELERATION)
                
                if TemporalPatterns.OSCILLATION in self.config.temporal_patterns and TemporalPatterns.INTERRUPTION in self.config.temporal_patterns:
                    if random.choice((True, False)):
                        selected_temporal_patterns.remove(TemporalPatterns.OSCILLATION)
                    else:
                        selected_temporal_patterns.remove(TemporalPatterns.INTERRUPTION)

                # Can be empty
                selected_temporal_patterns = random.choices(selected_temporal_patterns, k=random.randint(0, len(selected_temporal_patterns)))
                
                acceleration, deceleration, oscillation_period, interruption_period = 0.0, 0.0, 0.0, 0.0
                step_swap = 0.0
                
                if TemporalPatterns.OSCILLATION in selected_temporal_patterns:
                    oscillation_period = random.choice(list(range(self.config.oscillation_period.min, self.config.oscillation_period.max + 1)))
                elif TemporalPatterns.INTERRUPTION in selected_temporal_patterns:
                    interruption_period = random.choice(list(range(self.config.interruption_period.min, self.config.interruption_period.max + 1)))
                
                if TemporalPatterns.ACCELERATION in selected_temporal_patterns:
                    step /= 1.5
                    acceleration = 1.0 + random.randint(self.scale(self.config.acceleration_hundredth.min / 2), self.scale(self.config.acceleration_hundredth.max / 2)) / (self.scale_factor * 100)
                elif TemporalPatterns.DECELERATION in selected_temporal_patterns:
                    step *= 1.5
                    deceleration = 1.0 + random.randint(self.scale(self.config.acceleration_hundredth.min * 2), self.scale(self.config.acceleration_hundredth.max * 2)) / (self.scale_factor * 100)
                    deceleration = 1 / deceleration

                transformed_images, angles = [], []
                for i in range(self.config.context_size + current_batch_time):
                    if step != 0.0:
                        figure.rotate_z(step, point=scene.center_of_mass, inplace=True)
                        scene.plotter.render()

                    image = np.array(scene.plotter.screenshot())
                    transformed_images.append(torch.Tensor(self.transform(image)).to(device=self.device))
                    angles.append(step)

                    # Multiple patterns at a time
                    if TemporalPatterns.OSCILLATION in selected_temporal_patterns:
                        if (i + 1) % oscillation_period == 0:
                            step *= -1.0
                    elif TemporalPatterns.INTERRUPTION in selected_temporal_patterns:
                        if (i + 1) % interruption_period == 0:
                            step, step_swap = step_swap, step

                    if TemporalPatterns.ACCELERATION in selected_temporal_patterns:
                        step *= acceleration
                    elif TemporalPatterns.DECELERATION in selected_temporal_patterns:
                        step *= deceleration

                batch_images.append(torch.stack(transformed_images))
                batch_angles.append(torch.Tensor(angles))
                batch_temp_patterns.append(torch.Tensor([acceleration, deceleration, oscillation_period, interruption_period]))

            yield (
                torch.arange(start=1, end=current_batch_time + 1).to(device=self.device), 
                torch.stack(batch_images), 
                torch.stack(batch_angles), 
                torch.stack(batch_temp_patterns)
            )
