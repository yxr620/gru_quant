python train.py --end 2016-12-32
python train.py --end 2017-06-31
python train.py --end 2017-12-32
python train.py --end 2018-06-31
python train.py --end 2018-12-32
python train.py --end 2019-06-31
python train.py --end 2019-12-32
python train.py --end 2020-06-31
python train.py --end 2020-12-32
python train.py --end 2021-06-31
python train.py --end 2021-12-32
python train.py --end 2022-06-31
python train.py --end 2022-12-32
python train.py --end 2023-06-31

python prediction.py --start 2017-01-01 --end 2017-06-31 --model 2016-12-32 --type 1 --device cuda
python prediction.py --start 2017-07-01 --end 2017-12-32 --model 2017-06-31 --type 1 --device cuda
python prediction.py --start 2018-01-01 --end 2018-06-31 --model 2017-12-32 --type 1 --device cuda
python prediction.py --start 2018-07-01 --end 2018-12-32 --model 2018-06-31 --type 1 --device cuda
python prediction.py --start 2019-01-01 --end 2019-06-31 --model 2018-12-32 --type 1 --device cuda
python prediction.py --start 2019-07-01 --end 2019-12-32 --model 2019-06-31 --type 1 --device cuda
python prediction.py --start 2020-01-01 --end 2020-06-31 --model 2019-12-32 --type 1 --device cuda
python prediction.py --start 2020-07-01 --end 2020-12-32 --model 2020-06-31 --type 1 --device cuda
python prediction.py --start 2021-01-01 --end 2021-06-31 --model 2020-12-32 --type 1 --device cuda
python prediction.py --start 2021-07-01 --end 2021-12-32 --model 2021-06-31 --type 1 --device cuda
python prediction.py --start 2022-01-01 --end 2022-06-31 --model 2021-12-32 --type 1 --device cuda
python prediction.py --start 2022-07-01 --end 2022-12-32 --model 2022-06-31 --type 1 --device cuda
python prediction.py --start 2023-01-01 --end 2023-06-31 --model 2022-12-32 --type 1 --device cuda

@REM sep training
python train_sep.py --end 2016-12-32
python train_sep.py --end 2017-06-31
python train_sep.py --end 2017-12-32
python train_sep.py --end 2018-06-31
python train_sep.py --end 2018-12-32
python train_sep.py --end 2019-06-31
python train_sep.py --end 2019-12-32
python train_sep.py --end 2020-06-31
python train_sep.py --end 2020-12-32
python train_sep.py --end 2021-06-31
python train_sep.py --end 2021-12-32
python train_sep.py --end 2022-06-31
python train_sep.py --end 2022-12-32
python train_sep.py --end 2023-06-31

python prediction.py --start 2017-01-01 --end 2017-06-31 --model 2016-12-32 --type 2 --device cuda
python prediction.py --start 2017-07-01 --end 2017-12-32 --model 2017-06-31 --type 2 --device cuda
python prediction.py --start 2018-01-01 --end 2018-06-31 --model 2017-12-32 --type 2 --device cuda
python prediction.py --start 2018-07-01 --end 2018-12-32 --model 2018-06-31 --type 2 --device cuda
python prediction.py --start 2019-01-01 --end 2019-06-31 --model 2018-12-32 --type 2 --device cuda
python prediction.py --start 2019-07-01 --end 2019-12-32 --model 2019-06-31 --type 2 --device cuda
python prediction.py --start 2020-01-01 --end 2020-06-31 --model 2019-12-32 --type 2 --device cuda
python prediction.py --start 2020-07-01 --end 2020-12-32 --model 2020-06-31 --type 2 --device cuda
python prediction.py --start 2021-01-01 --end 2021-06-31 --model 2020-12-32 --type 2 --device cuda
python prediction.py --start 2021-07-01 --end 2021-12-32 --model 2021-06-31 --type 2 --device cuda
python prediction.py --start 2022-01-01 --end 2022-06-31 --model 2021-12-32 --type 2 --device cuda
python prediction.py --start 2022-07-01 --end 2022-12-32 --model 2022-06-31 --type 2 --device cuda
python prediction.py --start 2023-01-01 --end 2023-06-31 --model 2022-12-32 --type 2 --device cuda
