# """
# A test script for the CTRNN class.
# """
# import os
# import shutil
# import logging
# import traceback
# import os.path as osp

# from test_models import test_models


# if __name__ == '__main__':
#     # get the current file directory
#     current_file_dir = os.path.dirname(os.path.realpath(__file__))

#     # set up logging ===========================================
#     logger = logging.getLogger('my_logger')
#     logger.setLevel(logging.DEBUG)

#     # remove osp.join(current_file_dir, 'test.log')
#     if os.path.exists(osp.join(current_file_dir, 'test.log')):
#         os.remove(osp.join(current_file_dir, 'test.log'))
#     file_handler = logging.FileHandler(osp.join(current_file_dir, 'test.log'))
#     stream_handler = logging.StreamHandler()
#     logger.addHandler(file_handler)
#     logger.addHandler(stream_handler)
#     # ==========================================================

#     # test list ================================================
#     test_list = [
#         'test_base_rnn'
#     ]

#     # create a testing file for storing testing results
#     test_path = osp.join(current_file_dir, './test_temp')
#     if os.path.exists(test_path):
#         shutil.rmtree(test_path)
#     os.mkdir(test_path)
#     # ==========================================================

#     # test functions ===========================================
#     if 'test_base_rnn' in test_list:
#         logger.info('Testing base RNN...')
#         logger.info('='*80)
#         try:
#             test_models(logger)
#             logger.info('Base RNN Passed.')
#         except Exception as e:
#             # print error massage and line number
#             traceback.print_exc()
#             logger.error(f'Base RNN Failed. Error: {e}')
#     # ==========================================================

#     # remove the testing file ==================================
#     shutil.rmtree(test_path)
#     # ==========================================================
