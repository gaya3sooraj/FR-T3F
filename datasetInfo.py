db_info = {
    'orl': {
        'train_path': 'orl/traindata',
        'test_path': 'orl/testdata',
        'file_ext': '*.jpg',
        'd1' : 5600,
        'd2': 8,
        'd3' : 40,
        'name_pattern': lambda file: file.split('_')[1].split('.')[0]
        
    }
}
