#!/usr/bin/python3

import os
import sys

name_file = "./slice.txt"

with open(name_file) as file_names:
  print("Open names file: " + name_file)

  lines = file_names.readlines()

input_path_parent = "./src/nn/specs/"
output_path_parent = "../../test/"
output_all_jsTest = output_path_parent + "cts-all.js"

if os.path.exists(output_path_parent):
  cmd = "rm -r " + output_path_parent
  os.system(cmd)

if not os.path.exists(output_path_parent):
  os.makedirs(output_path_parent)

all_jsTest_file = open(output_all_jsTest, "w")
all_jsTest_file.write("describe('CTS', function() {\n")
all_jsTest_file.write("  const assert = chai.assert;\n")
all_jsTest_file.write("  const nn = navigator.ml.getNeuralNetworkContext();\n")
all_jsTest_file.close()

versions = os.listdir(input_path_parent)

for version in versions:
  input_path = input_path_parent + version + "/"
  output_path = output_path_parent + version + "/"

  if os.path.isdir(input_path):
    for line in lines:
      line = line.strip()

      input_file = input_path + line
      output_file_jsTest = output_path + line[:-6] + "js"

      if os.path.exists(input_file):
        if not os.path.exists(output_path):
          os.makedirs(output_path)

        cmd = "python3 ./src/test_generator.py " + input_file + " -js " + output_file_jsTest + " -a " + output_all_jsTest
        os.system(cmd)

all_jsTest_file = open(output_all_jsTest, "a+")
all_jsTest_file.write("});\n")
all_jsTest_file.close()

print("Generated all CTS tests in %s" % output_all_jsTest)
