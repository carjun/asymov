import os
import glob

path = '/content/drive/MyDrive/asymov_mt_ps102/reconstructions-simpler'
files = glob.glob(path + '/asymov_mt/*/*/*')      #return all mp4(s) in asymov_mt/naive*/*/*.mp4

# count = 0
for file in files:

  print('/'.join(file.split('/')[-4:-1]) + ' ........')

  new_dir_cmd = f"mkdir -p /content/drive/MyDrive/asymov_mt_ps102/reconstructions-simpler-converted/{'/'.join(file.split('/')[-4:-1])}"
  new_dir_path = f"/content/drive/MyDrive/asymov_mt_ps102/reconstructions-simpler-converted/{'/'.join(file.split('/')[-4:-1])}/{file.split('/')[-1]}"
  command = f'ffmpeg -i "{file}" -vcodec libx264 -acodec aac {new_dir_path}'

  os.system(new_dir_cmd)
  os.system(command)