import os
import pandas as pd


def valid_data(csv_file, data_dir, logger):

   if not os.path.isfile(csv_file):
      raise RuntimeError(f'csv file=[{csv_file}] does not exist.')
   if not os.path.isdir(data_dir):
      raise RuntimeError(f'data dir=[{data_dir}] does not exist.')

   df = pd.read_csv(csv_file)

   file_list = [file for file in os.listdir(data_dir) if file.endswith('.png')]

   logger.info(f'number of ct files from csv file ==> [{len(df)}]')
   logger.info(f'number of ct files in [{data_dir}] ==> [{len(file_list)}]')
   logger.info(f'validation result: number of missing files in [{data_dir}] = [{len(df) - len(file_list)}]')

   diff_df  = df.loc[~(df['filename'].isin(file_list))]
   if len(diff_df) != 0:
      filename_list = diff_df['filename'].values

      logger.error(f'------- [missing file list in [{data_dir}]] ---------------')
      for filename in filename_list:
         logger.error(filename)
      logger.error(f'--------------------------------------------------------------')

   else:
      logger.info(f'validation check is passed')


if __name__ == '__main__':

   valid_data('../dataset/dataset.csv', '../dataset/dataset_256/ct')










