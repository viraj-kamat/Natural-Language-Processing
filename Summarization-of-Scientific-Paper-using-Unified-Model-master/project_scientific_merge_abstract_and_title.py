import sys
import os





HIGHLIGHT = "@highlight"
abstract_files = ["tok.test.abstract.txt", "tok.train.abstract.txt", "tok.valid.abstract.txt"]
title_files = ["tok.test.title.txt", "tok.train.title.txt", "tok.valid.title.txt"]
abstract_dirs = ["abstract_dir/test", "abstract_dir/train", "abstract_dir/valid"]
prefix_list = ["test_input_", "train_input_", "valid_input_"]

def merge_and_create_files(input_dir):
  for (abstract_file, title_file, output_dir, prefix) in zip(abstract_files, title_files, abstract_dirs, prefix_list):
      abstract_reader = open(input_dir + "\\" + abstract_file, "r", encoding="utf8")
      title_reader = open(input_dir + "\\" + title_file, "r", encoding="utf8")
      print("abstract %s title %s" % (abstract_file, title_file))
      abstract = abstract_reader.readline()
      title = title_reader.readline()
      cnt = 0
      while abstract:
          cnt+=1
          if cnt > 100000:
              break
          with open(output_dir + "/" + prefix + str(cnt) + ".txt", "w", encoding="utf8") as writer:
              line = abstract + "\n" + HIGHLIGHT + "\n\n" + title
              writer.write(line)
              pass
          abstract = abstract_reader.readline()
          title = title_reader.readline()
          pass



# tok.test.abstract.txt
if __name__ == '__main__':
  if len(sys.argv) != 2:
    print("USAGE: python make_datafiles.py <scientific_abstract_text_dir>")
    sys.exit()
  input_dir = sys.argv[1]

  for dir in abstract_dirs:
    if not os.path.exists(dir): os.makedirs(dir)
  merge_and_create_files(input_dir)


