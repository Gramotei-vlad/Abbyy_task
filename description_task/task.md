# Требоания к проекту:
- Одна из трех задач: 
    - Part-of-speech tagging
    - Lemmatization
    - Grammatical error correction (or spelling)
- Как минимум 2 различных подхода
    - Пример: для POS-tagging: 1) flair embeddings  2) BERT + bpemb
    - Для каждого еще подобрать гиперпараметры
    - Всего 6+ экспериментов
- Результаты экспериментов в wandb:
    - Динамика лоссов на трейне/тесте
    - Динамика метрик по эпохам
    - Саммари по каждому эксперименту: лучшая метрика (эпоха) на тесте
- Корпус желательно (но необязательно) на русском
- Код оформлен и к нему есть README с описанием подходов и как что запускать (т.е. должны быть скрипты train и predict), инференс модели по чекпоинту. Пример для GEC:

 ```
 python predict.py --checkpoint model.ckpt --input "She see Tom is catched by policeman in park at last night."
 >>> She saw Tom caught by a policeman in the park last night.
 ```
- Задачу, подходы и корпус желательно согласовать заранее
- Сделать анализ ошибок лучшей модели
- Желательно все скинуть до конца семестра, чтобы успеть доделать