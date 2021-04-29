home=~/minyeong_workspace/vid2vid/datasets/face/desk


cd $home

for file in *.mp4
do
    mv $file/crop $file/img
done

