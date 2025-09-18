### Directory

-   dataset/{seq}/img1
    -   save per frame image of {seq}
-   gta_tracklets/{seq}.txt
    -   gta results of {seq}
-   tracklets_array
    -   numpy array format of tracker results
-   tracklets_vis
    -   save per frame bbox of trackelts
-   videos/{seq}

### Tool Scripts

-   count_unique_ids.py
-   count_tracklet_bboxes.py
-   video2img.py

### API

-   txt2array
-   split_tracklet
-   merge_tracklets
-   interpolate_tracklet
-   visualize_array
-   visualize_tracklets

### Steps for testing

-   put `{seq}.mp4` under `videos`
-   put gta results `{seq}.txt` under `gta_tracklets`
-   run `python video2img.py`, it will generate `{seq}.mp4` per frame image under `dataset/{seq}/img1`
-   run `python txt2array.py` to format the txt into 2D array (#tracklets \* #frames), it will save as `{seq}.npy` under `tracklets_array`
-   run `python visualize_tracklets.py`, it will generate per frame bbox for every tracklet, under `tracklets_vis`
-   (optional) run `python visualize_array.py` to visualize 2D array
-   run `python split_tracklet.py` to split the tracklet from specific frame, it will update `tracklets_array` and `tracklets_vis`
-   (optional) run `python visualize_array.py` to visualize 2D array
-   run `python merge_tracklets.py` to merge two trackelts. Note that two tracklets can't appear at the same frame
-   (optional) run `python visualize_array.py` to visualize 2D array
-   run `python interpolate_trackelt.py` to do interpolation when there are gap in a tracklet
-   (optional) run `python visualize_array.py` to visualize 2D array
