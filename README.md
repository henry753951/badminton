
# Badminton



## Dataset

#### Generat ball position dataset

```
  gen_dataset.py  # dataset-generator 
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `prue_csv_mode`   | `bool`| If True, the program will not output images. |
| `max_distacne`    | `int` |  The maximum distance threshold for detecting movement between consecutive frames. |
| `MaxTryToTrack`   | `int` | The maximum number of attempts to track an object in subsequent frames. |
| `dataset_dir`   | `string` | The directory where the ball dataset is stored. |



## Requirement



```
imutils==0.5.4
keras==2.10.0
numpy==1.24.3
pandas==2.0.1
piexif==1.1.3
Pillow==9.5.0
tensorflow==2.10.0
```

