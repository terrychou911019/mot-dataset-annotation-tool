### Directory

-   dataset/{seq}/img1
    -   save per frame image of {seq}
-   gta_tracklets/{seq}.txt
    -   gta results of {seq}
-   tracklets_array
    -   numpy array format of tracker results
-   videos/{seq}

### Steps

-   put `{seq}.mp4` under `videos`
-   run `python video2img.py`, it will generate `{seq}.mp4` per frame image under `dataset/{seq}/img1`
-   put gta results `{seq}.txt` under `gta_tracklets`
-   run `python txt2array.py` to format the txt into 2D array (#tracklets \* #frames), it will save as `{seq}.npy` under `tracklets_array`
-   (optional) run `python visualize_array.py` to visualize 2D array
-   `python visualize_tracklets.py`, it will generate per frame bbox for every tracklet, under `tracklets_vis`
-   run `python split_tracklet.py` to split the tracklet from specific frame, it will update `tracklets_array` and `tracklets_vis`
