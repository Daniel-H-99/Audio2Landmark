home = /home/server25/minyeong_workspace/vid2vid
target_dir = datasets/face/test_keypoints/keypoints

cd $target_dir

for file in *
do
 mv $file ${file:1}
done

cd $home
