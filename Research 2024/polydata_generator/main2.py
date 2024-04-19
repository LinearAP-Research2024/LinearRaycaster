ret = ""
def read_file_as_list(file_path):
    try:
        with open(file_path, 'r') as file:
            # Read lines from the file and remove newline characters
            lines = [line.strip() for line in file.readlines()]
            return lines
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

full = read_file_as_list("polydata2.txt")
for i in range(57):
    ret+="\n"
    ret+="255={0.0,0.0,1.0,0.0,0.0,0.0}-"
    ret+=full[i]

for i in range(54):
    ret+="\n"
    ret+="*203={0.0,0.0,1.0,0.0,0.0,0.0}-"
    ret+=full[i + 57]

for i in range(17 + 21):
    ret+="\n"
    ret+="*52={0.0,0.0,1.0,0.0,0.0,0.0}-"
    ret+=full[i + 54+57]

for i in range(19 + 5 * 11):
    ret+="\n"
    ret+="25={0.0,0.0,1.0,0.0,0.0,0.0}-"
    ret+=full[i + 54+57+17+21]

    
def write_string_to_file(file_path, content):
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        print(f"String successfully written to '{file_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

write_string_to_file("polydata.txt", ret)