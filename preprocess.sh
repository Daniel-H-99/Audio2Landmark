src_dir=data_root/lof
tgt_dir=data_preprocessed/lof
home=~/minyeong_workspace

# start_time="00:00:00"
# end_time="00:03:00"

cd $home/$src_dir

for file in *.mp4
do
    mkdir -p $home/$tgt_dir/$file
    mkdir -p $home/$tgt_dir/$file/full
    mkdir -p $home/$tgt_dir/$file/crop
    ffmpeg -hide_banner -y -i $file -r 25 $home/$tgt_dir/$file/full/%05d.png
    python $home/vid2vid/util/crop_portrait.py --data_dir $home/$tgt_dir/$file --crop_level 2 --vertical_adjust 0.6
    rm -rf $home/$tgt_dir/$file/full
    ffmpeg -loglevel panic -y -i $file -strict -2 $home/$tgt_dir/$file/audio.wav
done

cd $home
