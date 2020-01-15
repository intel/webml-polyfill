#!/usr/bin/python3

import os
import argparse
import json

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("-o", "--output", help = "[option] '-o [output all-file in relative directory]', create all test file", default = "-")
  parser.add_argument("-t", "--transfer", help = "[option] '-t [transfer relative directory]', transfer nn test file", default = "-")
  parser.add_argument("-c", "--cts", help = "[option] '-c [cts relative directory]', include cts test file", default = "-")
  parser.add_argument("-s", "--supplement", help = "[option] '-s [supplement relative directory]', include supplement test file", default = "-")
  parser.add_argument("-p", "--plus", help = "[option] '-p [plus relative directory]', include plus test file", default = "-")

  args = parser.parse_args()

  return (args.output, args.transfer, args.cts, args.supplement, args.plus)

def del_file(path, flag):
  names = os.listdir(path)

  for name in names:
    file_path = os.path.join(path, name)

    if os.path.isdir(file_path):
      del_file(file_path, True)
    else:
      os.remove(file_path)

  if flag:
    os.rmdir(path)

# scan test case directory
def get_file_names(ipath, suffixName):
  file_names_dict = dict()

  names = os.listdir(ipath)

  for name in names:
    path_or_file = os.path.join(ipath, name)

    if os.path.isfile(path_or_file):
      if name[-3:] == suffixName:
        file_names_dict[name] = path_or_file
    else:
      tmp_dict = get_file_names(path_or_file, suffixName)
      file_names_dict.update(tmp_dict)

  return file_names_dict

# transfer nn test case to js test case
def transfer(ipath, opath, version_names):
  for (version, names) in version_names.items():
    input_dir = os.path.join(ipath, version)
    output_dir = os.path.join(opath, version)

    assert os.path.exists(input_dir), "input directory is not exist: %s"%input_dir

    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    for name in names:
      input_file = os.path.join(input_dir, name)
      output_file = os.path.join(output_dir, name[:-6] + "js")

      assert os.path.exists(input_file), "input file is not exist: %s"%input_file

      cmd = "python3 ./src/cts_generator.py " + input_file +\
            " -js " + output_file
      os.system(cmd)

# create all test case file
def create(opath, file_dict, file_list, describe):
  with open(opath, "w") as all_jsTest_file:
    all_jsTest_file.write("describe('" + describe + "', function() {\n")
    all_jsTest_file.write("  this.timeout(20000);\n")
    all_jsTest_file.write("  const assert = chai.assert;\n")
    all_jsTest_file.write("  const nn = navigator.ml.getNeuralNetworkContext();\n")
    all_jsTest_file.write("\n")

  for (file_num, file_name) in enumerate(file_list):
    with open(file_dict.get(file_name), "r") as file_read:
      file_text = file_read.readlines()

      for (line_num, line_text) in enumerate(file_text):
        if line_num in range(4, len(file_text) - 1):
          with open(opath, "a+") as all_jsTest_file:
            all_jsTest_file.write(line_text)

      if file_num in range(len(file_list) - 1):
        with open(opath, "a+") as all_jsTest_file:
          all_jsTest_file.write("\n")

  with open(opath, "a+") as all_jsTest_file:
    all_jsTest_file.write("});\n")

if __name__ == "__main__":
  (args_all, args_transfer, args_cts, args_supplement, args_plus) = get_args()

  path_root = "./"
  output_path_root = "./output"
  if not os.path.exists(output_path_root):
    os.makedirs(output_path_root)

  (output_file_all_path, output_file_all_name) = os.path.split(args_all)
  if output_file_all_path == "":
    output_file_all = os.path.join(path_root, args_all)
  else :
    output_file_all = args_all

  describeString = "CTS"

  file_dict_all = dict()

  if not args_transfer == "-":
    output_path_transfer_clear = os.path.join(output_path_root, "cts")
    if os.path.exists(output_path_transfer_clear):
      del_file(output_path_transfer_clear, False)
    else:
      os.makedirs(output_path_transfer_clear)

    output_path_transfer = os.path.join(output_path_transfer_clear, "specs")
    support_cts_file = "./slice.json"

    with open(support_cts_file) as cts_file:
      cts_file_names = json.load(cts_file)
      print ("Open support cts file: " + support_cts_file + "\n")

    print ("transfer nn test case to js test case....\n")
    transfer(args_transfer, output_path_transfer, cts_file_names)

  if not args_cts == "-":
    print ("scan test cts directory....")
    cts_file_dict = get_file_names(args_cts, ".js")
    file_dict_all.update(cts_file_dict)

  if not args_supplement == "-":
    print ("scan test supplement directory....")

    describeString = "CTS Supplement Test"

    supplement_file_dict = get_file_names(args_supplement, ".js")
    file_dict_all.update(supplement_file_dict)

  if not args_plus == "-":
    print ("scan test plus directory....")
    plus_file_dict = get_file_names(args_plus, ".js")
    file_dict_all.update(plus_file_dict)

  if not args_all == "-":
    skip_files = ["cts-all.js", "cts_supplement-all.js", "conv_65_65_96.js"]
    (args_all_path, args_all_name) = os.path.split(args_all)
    skip_files.append(args_all_name)

    for name in skip_files:
      if name in file_dict_all:
        print ("skip creating file name: %s"%file_dict_all[name])
        del file_dict_all[name]

    print ("reordering by name....")
    file_list = sorted(file_dict_all.keys())

    print ("create all test case file: %s"%str(output_file_all))
    create(output_file_all, file_dict_all, file_list, describeString)

  if not args_transfer == "-":
    if not args_all == "-":
      print ("transfer and create all files are completed")
    else :
      print ("transfer all files are completed")
  else :
    if not args_all == "-":
      print ("create all files are completed")
    else :
      print ("nothing to do!! get info with 'npm run info' command")
