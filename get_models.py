# Copyright 2020 Helga Svala 

def get_model(model_type):
  """Get the model files from CLARIN"""
  try:
    os.mkdir(model_type)
    print(f"Created {model_type}-folder")
  except:
    if os.path.exists(model_type):
        pass
  base_url = "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/49/"
  to_download = []
  if model_type == "standard":
    to_download.extend(
        [
            (f"{base_url}isl-model.pcl", model_type + "/model.pcl"),
            (f"{base_url}vocabulary", model_type + "/vocabulary"),
            (f"{base_url}punctuations", model_type + "/punctuations"),
        ]
    )
  elif model_type == "north":
    to_download.extend(
      [
          (f"{base_url}pytorch_model.bin", model_type + "/pytorch_model.bin"),
          (f"{base_url}vocab.txt", model_type + "/vocab.txt"),
          (f"{base_url}config.json", model_type + "/config.json"),
          (
              f"{base_url}tokenizer_config.json",
              model_type + "/tokenizer_config.json",
          ),
          (
              f"{base_url}special_tokens_map.json",
              model_type + "/special_tokens_map.json",
          ),
      ]
    )
  else:
    sys.stderr.write("There is no matching model to the argument.")

  for download_args in to_download:
    try:
        download_file(*download_args)
    except:
        logging.error("Could not download file.")
        sys.exit(0)

