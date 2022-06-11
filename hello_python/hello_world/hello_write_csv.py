import csv

with open('../assets/protagonist.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter =';')
    writer.writerow(["SN", "Movie", "Protagonist"])
    writer.writerow([1, "Lord of the Rings", "Frodo Baggins"])
    writer.writerow([2, "Harry Potter", "Harry Potter"])