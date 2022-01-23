from datetime import datetime

# date_example = "8/6/2021, 05:54:8"
# date_format = datetime.datetime.strptime(date_example,
#                                          "%m/%d/%Y, %H:%M:%S")
# unix_time = datetime.datetime.timestamp(date_format)
# print(unix_time)



test = "2019-12-06 11:12:22"
test3 = "11.12.2019 11:12"

ami = datetime.strptime(test, "%Y-%m-%d %H:%M:%S")

ger = datetime.strptime(test3, "%d.%m.%Y %H:%M")

unix_timeUS = datetime.timestamp(ami)
unix_timeGER = datetime.timestamp(ger)
print(unix_timeUS)
print(unix_timeGER)


def func(s):

    if '-' in s:
        return datetime.timestamp(datetime.strptime(s, "%Y-%m-%d %H:%M:%S"))
    else:
        return datetime.timestamp(datetime.strptime(s, "%d.%m.%Y %H:%M"))