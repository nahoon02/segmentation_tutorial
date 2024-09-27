import datetime


def convert_second_to_time(seconds: float) -> str:
    """ convert second format to day, hours, minutes and seconds format """

    delta = datetime.timedelta(seconds=seconds)
    days = delta.days
    _sec = delta.seconds
    (hours, minutes, seconds) = str(datetime.timedelta(seconds=_sec)).split(':')
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)

    result = []
    if days >= 1:
        result.append(str(days) + ' [days]')
    if hours >= 1:
        result.append(str(hours) + ' [hours]')
    if minutes >= 1:
        result.append(str(minutes) + ' [minutes]')
    if seconds >= 1:
        result.append(str(seconds) + ' [seconds]')

    return ' '.join(result)
