# Convert image messages to rgb format

##
1. Put the rosbag file at the same directory
2. Run the python script with
```python
python convert_rosbag.py bagfile_name --split
```

--split is optional and would split the rosbag to several bag files with length 60 seconds each when provided.
The topic name in the bag file is hardcoded and would need to be changed for different bag file
