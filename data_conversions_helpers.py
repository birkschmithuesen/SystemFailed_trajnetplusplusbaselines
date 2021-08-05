from trajnetdataset.convert_pharus import main

def pharus_convert(pharus_file_path, output_path):
    args = ["--input_file", pharus_file_path, "--output_path", output_path]
    main(args)
