# Get tsla data for all intervals
# grep  "After Hours" to the first "</div>"
# user input k for if, anything else for else
# If file is called by jinx.py, then the user input is not needed.
# If file is called by the user, then the user input is needed.
# elif [ "$choice" = "k" ]; then
#     grep 'After Hours' tsla
# elif [ "$choice" = "d" ]; then
#     grep '(?=</div>)' tsla

# Store data in sqlite3 database tsla.db using a function
# Create a table with the following columns: date, time, price, after_hours, interval, and volume, and insert the data into the table.
# function create_table {
    # sqlite3 tsla.db "CREATE TABLE tsla (date TEXT, time TEXT, price REAL, after_hours REAL, interval TEXT, volume INTEGER);"
# }
# Create a function to insert data into the table.
# Create a function to display the data in the table.
# Create a function to delete the table.
# Create a function to delete the database.

