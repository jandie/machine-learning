import re


def get_log(file_name):
    log = []

    with open("./" + file_name) as the_file:
        first_line = True

        for line in the_file:
            if not first_line:
                line = line.replace('\n', '')
                line = line.replace(' ', '')
                log.append(re.split(',', line))

            first_line = False

    return log

get_log()