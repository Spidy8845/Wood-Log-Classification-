import Augmentor

week_list = ["week 1", "week 2", "week 3", "week 4", "week 5", "week 6"]

for week in week_list:    
    path = fr"F:\360 digiTMG\project2\Dataset\{week}"
    print(f"Processing directory: {path}")
    p = Augmentor.Pipeline(path)
    p.zoom(probability=0.3, min_factor=0.8, max_factor=1.5)
    p.flip_top_bottom(probability=0.4)
    #p.random_brightness(probability=0.3, min_factor=0.3, max_factor=1.2)
    p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
    p.flip_left_right(probability=0.5)
    p.rotate_random_90(probability=0.7)
    p.skew_top_bottom(probability=0.5, magnitude=0.8)
    p.skew_tilt(probability=0.6, magnitude=0.7)
    p.sample(300)
