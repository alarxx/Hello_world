import datetime


delta = datetime.timedelta(hours=10, minutes=10, seconds=20)

total_seconds = delta.total_seconds()

minutes = int(total_seconds // 60)
seconds = int(total_seconds % 60)

print(f'{minutes}:{seconds}')