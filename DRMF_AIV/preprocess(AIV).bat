#Parse raw review datasets
python ./reviews_processor.py

#Split and generate datasets for training, validation and test
python ./run.py -d ./test/AIV/1 -a ./test/AIV/1 -c True -r ./data/Amazon_Instant_Video_5/ratings.dat -i ./data/Amazon_Instant_Video_5/item_content.dat -u ./data/Amazon_Instant_Video_5/user_content.dat -m 1
python ./run.py -d ./test/AIV/2 -a ./test/AIV/2 -c True -r ./data/Amazon_Instant_Video_5/ratings.dat -i ./data/Amazon_Instant_Video_5/item_content.dat -u ./data/Amazon_Instant_Video_5/user_content.dat -m 1
python ./run.py -d ./test/AIV/3 -a ./test/AIV/3 -c True -r ./data/Amazon_Instant_Video_5/ratings.dat -i ./data/Amazon_Instant_Video_5/item_content.dat -u ./data/Amazon_Instant_Video_5/user_content.dat -m 1
python ./run.py -d ./test/AIV/4 -a ./test/AIV/4 -c True -r ./data/Amazon_Instant_Video_5/ratings.dat -i ./data/Amazon_Instant_Video_5/item_content.dat -u ./data/Amazon_Instant_Video_5/user_content.dat -m 1
python ./run.py -d ./test/AIV/5 -a ./test/AIV/5 -c True -r ./data/Amazon_Instant_Video_5/ratings.dat -i ./data/Amazon_Instant_Video_5/item_content.dat -u ./data/Amazon_Instant_Video_5/user_content.dat -m 1

set/p xxxx= >nul
echo %xxxx%