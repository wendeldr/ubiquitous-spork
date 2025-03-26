
import re
import csv

# Define the regex pattern
pattern = r"Time taken for \[None: (.*?)\]: (\d+\.\d+)s"

# Read the input text
with open("/home/dan/data/connectivity/pyspi_testing/timing/timings.txt", "r") as file:
    content = file.read()

# Find all matches
matches = re.findall(pattern, content)

# Save matches to a CSV file
output_file = "timings_output.csv"
with open(output_file, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write the header
    csv_writer.writerow(["Process", "Time (s)"])
    # Write the rows
    csv_writer.writerows(matches)

print(f"Data saved to {output_file}")
