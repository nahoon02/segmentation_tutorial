# mount drive

        import os
        from google.colab import drive
        import sys
        drive.mount('/content/drive/')
        my_path = '/content/package'        
        sys.path.append(my_path)

# install package

      !pip install --target=$my_path yacs

# change dir

      %cd /content/drive/MyDrive/Colab Notebooks
      !mkdir segmentation
      %cd /content/drive/MyDrive/Colab Notebooks/segmentation

# git clone

      !mkdir segmentation_tutorial
      !git clone https://github.com/nahoon02/segmentation_tutorial.git


# unzip dataset file


      !unzip -qq '/content/drive/MyDrive/Colab Notebooks/segmentation/segmentation_tutorial/dataset/dataset_256/mask.zip' -d '/content/drive/MyDrive/Colab Notebooks/segmentation/segmentation_tutorial/dataset/dataset_256'
      !unzip -qq '/content/drive/MyDrive/Colab Notebooks/segmentation/segmentation_tutorial/dataset/dataset_256/ct.zip' -d '/content/drive/MyDrive/Colab Notebooks/segmentation/segmentation_tutorial/dataset/dataset_256'


# run train.py

      %cd /content/drive/MyDrive/Colab Notebooks/segmentation/segmentation_tutorial/main
      %run train.py

      
        