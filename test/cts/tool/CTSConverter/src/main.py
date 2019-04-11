#!/usr/bin/python3

import os
import argparse

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("-a", "--all", help = "[option] '-a [file name]', create all test file", default = "-")
  parser.add_argument("-t", "--transfer", help = "[option] '-t [transfer directory]', transfer nn test file", default = "-")
  parser.add_argument("-c", "--cts", help = "[option] '-c [cts directory]', include cts test file", default = "-")
  parser.add_argument("-s", "--supplement", help = "[option] '-s [supplement directory]', include supplement test file", default = "-")
  parser.add_argument("-p", "--plus", help = "[option] '-p [plus directory]', include plus test file", default = "-")

  args = parser.parse_args()

  return (args.all, args.transfer, args.cts, args.supplement, args.plus)

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
def transfer(ipath, opath, names):
  dirs = os.listdir(ipath)

  for dir in dirs:
    idir = os.path.join(ipath, dir)
    odir = os.path.join(opath, dir)

    if os.path.isdir(idir):
      os.makedirs(odir)
      transfer(idir, odir, names)
    else:
      for name in names:
        name = name.strip()

        if name == dir:
          input_file = idir
          output_file = os.path.join(opath, dir[:-6] + "js")
          cmd = "python3 ./src/test_generator.py " + input_file +\
                " -js " + output_file
          os.system(cmd)

# create all test case file
def create(opath, file_dict, file_list, describe):
  with open(opath, "w") as all_jsTest_file:
    all_jsTest_file.write("describe('" + describe + "', function() {\n")
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

  output_path_root = "./output"
  if not os.path.exists(output_path_root):
    os.makedirs(output_path_root)

  output_file_all = os.path.join(output_path_root, args_all)

  describeString = "CTS"

  file_dict_all = dict()

  if not args_transfer == "-":
    output_path_transfer = os.path.join(output_path_root, "cts")
    if os.path.exists(output_path_transfer):
      del_file(output_path_transfer, False)
    else:
      os.makedirs(output_path_transfer)

    support_cts_file = "./slice.txt"

    with open(support_cts_file) as cts_file:
      cts_file_names = cts_file.readlines()
      print ("Open support cts file: " + support_cts_file + "\n")

    print ("transfer nn test case to js test case....\n")
    transfer(args_transfer, output_path_transfer, cts_file_names)

  if not args_cts == "-":
    print ("scan test cts directory....\n")
    cts_file_dict = get_file_names(args_cts, ".js")
    file_dict_all.update(cts_file_dict)

  if not args_supplement == "-":
    print ("scan test supplement directory....\n")

    describeString = "CTS Supplement Test"

    supplement_file_dict = get_file_names(args_supplement, ".js")
    file_dict_all.update(supplement_file_dict)

  if not args_plus == "-":
    print ("scan test plus directory....\n")
    plus_file_dict = get_file_names(args_plus, ".js")
    file_dict_all.update(plus_file_dict)

  if not args_all == "-":
    print ("reordering by name....\n")
    file_list = sorted(file_dict_all.keys())

    print ("create all test case file....\n")
    create(output_file_all, file_dict_all, file_list, describeString)

  print ("transfer and create all files are completed")
