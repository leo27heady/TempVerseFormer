from datetime import datetime


def create_timestamp() -> str:
    now = datetime.now()
    return now.strftime("%Y_%m_%d %H_%M_%S")
