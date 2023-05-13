
# BadmintonAI
## Requirement
- Python 3.10
- OpenCV
- requirements.txt

## Ball predit
```
  ball_pred.py
```
## Court predit
```
  court_pred.py
```


## Dataset
#### Generat ball position dataset

```
  tools/gen_dataset.py  # dataset-generator 
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `prue_csv_mode`   | `bool`| If True, the program will not output images. |
| `max_distacne`    | `int` |  The maximum distance threshold for detecting movement between consecutive frames. |
| `MaxTryToTrack`   | `int` | The maximum number of attempts to track an object in subsequent frames. |
| `dataset_dir`   | `string` | The directory where the ball dataset is stored. |





