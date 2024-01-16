#!/usr/bin/env python
# coding: utf-8

import os
import torch
import configparser
from datetime import datetime
from model.network import Network

config = configparser.ConfigParser()
config.read('./ATGCNet.config')

repeat_time = int(config["train"]["repeat_time"])
start_epoch = int(config["train"]["start_epoch"])
epochs = int(config["train"]["epochs"])
patient = int(config["train"]["patient"])

network = Network(config)

if network.mode == 'train_test':
    for _ in range(repeat_time):
        min_val_loss = float('inf')
        start = datetime.now()
        for epoch in range(start_epoch, epochs):
            network.logger.info('epochs [{}/{}]'.format(epoch + 1, epochs))
            starttime = datetime.now()
            # training
            train_loss, train_batch_cnt = network.train()
            # validate
            validate_loss, validate_batch_cnt = network.validate()
            # saving best validated model
            if min_val_loss > validate_loss:
                network.save(epoch, validate_loss)
                min_val_loss = validate_loss       
                early_stop_counter = 0
            endtime = datetime.now()
            network.logger.info('cost:{} s train loss:{} val_loss:{}'.format(
                (endtime - starttime).seconds,
                train_loss/train_batch_cnt,
                validate_loss/validate_batch_cnt
            ))
            early_stop_counter += 1
            if early_stop_counter >= patient:
                logger.info('Early Stop Training')
                break
        end = datetime.now()
        network.logger.info('Finished Training')
        network.logger.info('Total cost:{} s'.format((end-start).seconds))
        network.test()

if network.mode == 'test':
    network.test()
