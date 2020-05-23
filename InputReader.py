import csv

def get_dataset():
    # read file to my_list
    with open('dataset1.csv', newline='') as f:
        reader = csv.reader(f)
        my_list = list(reader)

    # fix first line
    my_list[0][0] = my_list[0][0][3:]

    # remove elements with missing features
    for line in my_list:
        if '?' in line:
            my_list.remove(line)

    # change 'R' to 1 and 'N' to 0
    for line in my_list:
        if line[1] == 'R':
            line[1] = 1
        else:
            line[1] = -1

    # fix types and round numbers to 4 digits after dot (3.1123231 -> 3.1123)
    for line in my_list:
        for i in range(2, len(line)):
            if isinstance(line[i], str):
                line[i] = float(line[i])
                line[i] = round(line[i], 4)
            else:
                line[i] = float(line[i])

    return my_list

# for line in my_list:
#     print(line)
# print(len(my_list))
