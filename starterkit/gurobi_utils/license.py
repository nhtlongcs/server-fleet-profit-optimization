def load_wsl_lic(lic_path='gurobi.lic'):
    # # Initialize an empty dictionary to hold the key-value pairs
    license_dict = {}
    with open(lic_path, 'r') as file:
        for line in file:
            # Strip any leading/trailing whitespace (including newline characters)
            line = line.strip()
            # Split the line into key and value
            if '=' in line:
                key, value = line.split('=', 1)  # Split on the first '='
                # Convert LICENSEID to an integer
                if key == 'LICENSEID':
                    license_dict[key] = int(value)
                else:
                    license_dict[key] = value
    return license_dict