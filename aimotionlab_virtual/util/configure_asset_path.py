import argparse
import os
import re

def set_path_for_assets(asset_path):

    directory=asset_path+"/xml_models/"

    print(directory)
    # Define the regular expression pattern to find ".." symbols
    pattern = r'\.\.'

    # Iterate through XML files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.xml'):
            filepath = os.path.join(directory, filename)

            # Read the XML file
            with open(filepath, 'r') as file:
                xml_content = file.read()

            # Replace all occurrences of ".." symbols with a replacement string (e.g., "")
            modified_content = re.sub(pattern, asset_path, xml_content)

            # Write the modified content back to the same file
            with open(filepath, 'w') as file:
                file.write(modified_content)
        print(f"Asset path consifred for file {filename}")

def main():
    parser = argparse.ArgumentParser(description="Asset path configurator")
    
    # Define the command-line arguments
    parser.add_argument('--path', help='Path of the assets directory', required=True)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values of the arguments
    asset_path = args.path
    set_path_for_assets(asset_path=asset_path)


if __name__ == "__main__":
    main()


