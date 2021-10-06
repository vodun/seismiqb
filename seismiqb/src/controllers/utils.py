""" Utilities to work with controllers and their logs. """
from collections import OrderedDict
from datetime import datetime
from ast import literal_eval



def line_to_time(line):
    """ Extract timestamp from a line, created by log. """
    time = line.split('    ')[0].split(',')[0]
    time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    return time


def log_to_dict(log_path):
    """ Parse controller log into a dictionary. """
    #pylint: disable=too-many-branches
    dct = OrderedDict()

    with open(log_path, encoding='utf-8') as file:
        for i, line in enumerate(file):
            if i == 0:
                dct['Начало вычислений'] = line_to_time(line)
                dct['Создание базового контроллера'] = line_to_time(line)

            # Get all the timestamps
            if line.startswith('20'):
                if 'Interpolator' in line:
                    if 'Initialized Interpolator' in line:
                        dct['Начало этапа: детекция'] = line_to_time(line)
                    if 'Created dataset' in line:
                        dct['Детекция: датасет (куб и каркас) загружены'] = line_to_time(line)
                    if 'Train run' in line:
                        dct['Детекция: начало обучения модели'] = line_to_time(line)
                    if 'Trained for ' in line:
                        dct['Детекция: конец обучения модели'] = line_to_time(line)
                    if 'Starting' in line and 'inference' in line:
                        dct['Детекция: начало процедуры предсказания'] = line_to_time(line)
                    if 'Inference done in' in line:
                        dct['Детекция: конец процедуры предсказания'] = line_to_time(line)

                    # The last mention of interpolator
                    if 'Dumped' not in line and 'Finished exp' not in line:
                        dct['Конец этапа: детекция'] = line_to_time(line)

                if 'Extender' in line:
                    if 'Initialized Extender' in line:
                        dct['Начало этапа: продление'] = line_to_time(line)
                    if 'Train run' in line:
                        dct['Продление: начало обучения модели'] = line_to_time(line)
                    if 'Trained for ' in line:
                        dct['Продление: конец обучения модели'] = line_to_time(line)
                    if 'Inference started for' in line:
                        dct['Продление: начало процедуры предсказания'] = line_to_time(line)
                    if 'Total points added' in line:
                        dct['Продление: конец процедуры предсказания'] = line_to_time(line)

                    # The last mention of extender
                    dct['Конец этапа: продление'] = line_to_time(line)
                dct['Конец вычислений'] = line_to_time(line)
            # Get all the resulting metrics
            if 'coverages ->' in line:
                dct['Покрытие горизонта, %'] = literal_eval(line.split('coverages ->')[1].strip())[-1]
            if 'window_rates ->' in line:
                dct['Близость к каркасу, %'] = literal_eval(line.split('window_rates ->')[1].strip())[-1]
            if 'phases ->' in line:
                dct['Среднее отклонение от медианной фазы'] = literal_eval(line.split('phases ->')[1].strip())[-1]
            if 'corrs ->' in line:
                dct['Метрика качества предсказания, [-1, 1]'] = literal_eval(line.split('corrs ->')[1].strip())[-1]

    dct = {
        'Детекция: общее время, с' : (dct['Конец этапа: детекция'] -
                                      dct['Начало этапа: детекция']).seconds,
        'Детекция: обучение модели, с' : (dct['Детекция: конец обучения модели'] -
                                          dct['Детекция: начало обучения модели']).seconds,
        'Детекция: инференс, с' : (dct['Детекция: конец процедуры предсказания'] -
                                   dct['Детекция: начало процедуры предсказания']).seconds,
        'Продление: общее время, с' : (dct['Конец этапа: продление'] -
                                       dct['Начало этапа: продление']).seconds,
        'Продление: обучение модели, с' : (dct['Продление: конец обучения модели'] -
                                           dct['Продление: начало обучения модели']).seconds,
        'Продление: инференс, с' : (dct['Продление: конец процедуры предсказания'] -
                                    dct['Продление: начало процедуры предсказания']).seconds,
        **dct
    }
    return dct
