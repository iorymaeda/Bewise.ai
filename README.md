# bewise.ai test task

Тестовое задание bewise.ai на позицию Junior Data Scientist (NLP)

### Задачи, которые должен выполнять скрипт:

1. Извлекать реплики с приветствием – где менеджер поздоровался.
2. Извлекать реплики, где менеджер представил себя.
3. Извлекать имя менеджера.
4. Извлекать название компании.
5. Извлекать реплики, где менеджер попрощался.
6. Проверять требование к менеджеру: «В каждом диалоге обязательно необходимо поздороваться и попрощаться с клиентом»

### Решение

Моё решение - это deeppavlov NER, куча процессинга и деплой всего этого дела в rest-api, для дальнешей возможности обработки других диалогов.
К сожалению NER deeppavlov не выдаёт скоры, поэтому адекватно достать имена не получилось, возможно использоваться другими подходами - регулярными выражениями например

### Disclaimer

Во время выполнения задания у меня возникла мысль, что роль менеджера и клиента в тестовых данных перепутана и дальнейшие вычисления происходили для клиента

### 🔨 Установка

##### Конфигурация

в [.env](.env) внести следующие переменные окружения:
`DP_COMPONENTS_VOLUME` - путь по которому сохранятся deeppavlov модели
`DP_VENV_VOLUME` - путь по которому сохранится deeppavlov python окружение
`LOCAL_PORT` - порт на локальной машине по которому можно будет обраться к серверу

##### Сборка и развёртывание проекта

В директории с проектом:

```
docker-compose up --build
```

### Пользование

Вручную загрузить файл в браузере:

```
http://localhost:{порт}/docs#/default/
```

Python код:

```python
>>> import requests

>>> url = "http://localhost:8300/process_csv/"
>>> resp = requests.post(url, files = {'csv_file': open('test_data.csv', 'rb')})
>>> resp.json()
# Выдаёт словарь с ключами равными `dlg_id` из csv
# greeting - менеджер попревествовался
# farewell - менеджер попрощался
# farewell_text - реплика с прощанием
# greeting_text - реплика с приветсвием 
# name_text - реплика где менеджер представи себя
# PER_name - имя менеджера
# ORG_name - имя организации 
# is_polite - менеджер и поздаровался, и попращался
{0: {'greeting': True,
  'farewell': True,
  'farewell_text': 'Всего хорошего до свидания',
  'greeting_text': 'Алло здравствуйте',
  'name_text': 'Меня зовут ангелина компания диджитал бизнес звоним вам по поводу продления лицензии а мы с серым у вас скоро срок заканчивается',
  'PER_name': ['ангелина'],
  'ORG_name': ['компания диджитал бизнес'],
  'is_polite': True},
 1: {'greeting': True,
  'farewell': True,
  'farewell_text': 'До свидания',
  'greeting_text': 'Алло здравствуйте',
  'name_text': 'Меня зовут ангелина компания диджитал бизнес звоню вам по поводу продления а мы сели обратила внимание что у вас срок заканчивается',
  'PER_name': ['ангелина'],
  'ORG_name': ['компания диджитал бизнес'],
  'is_polite': True},
 2: {'greeting': True,
  'farewell': False,
  'farewell_text': '',
  'greeting_text': 'Алло здравствуйте',
  'name_text': 'Меня зовут ангелина компания диджитал бизнес звоню вам по поводу продления лицензии а мастера мы с вами сотрудничали по видео там',
  'PER_name': ['ангелина'],
  'ORG_name': ['компания диджитал бизнес'],
  'is_polite': False},
 3: {'greeting': True,
  'farewell': True,
  'farewell_text': 'Угу все хорошо да понедельника тогда всего доброго',
  'greeting_text': 'Алло дмитрий добрый день',
  'name_text': 'Добрый меня максим зовут компания китобизнес удобно говорить',
  'PER_name': [' дмитрий', 'максим зовут', 'дмитрий', 'угу'],
  'ORG_name': ['китобизнес'],
  'is_polite': True},
 4: {'greeting': False,
  'farewell': True,
  'farewell_text': 'Во вторник все ну с вами да тогда до вторника до свидания',
  'greeting_text': '',
  'name_text': '',
  'PER_name': [],
  'ORG_name': [],
  'is_polite': False},
 5: {'greeting': False,
  'farewell': True,
  'farewell_text': 'Ну до свидания хорошего вечера',
  'greeting_text': '',
  'name_text': '',
  'PER_name': ['анастасия', 'угу'],
  'ORG_name': [],
  'is_polite': False}}
```