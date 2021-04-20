import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
import telebot
from tokenstorage import token
from random import choice
#Инициализация бота
bot = telebot.TeleBot(token)


def image_loader(image_name):
    loader = transforms.Compose([
    #Приводим к размеру 224*224, поскольку сеть обучалась на тензорах данного размера
    transforms.Resize((224)),
    transforms.ToTensor(),
    #Нормализация 
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

model = models.resnet152()
for param in model.parameters():
    param.requires_grad = False

# последний слой с двумя выходами по количеству классов в классификации
model.fc = torch.nn.Linear(model.fc.in_features, 2)
# подгружаем веса нейросети, обученной на Kaggle.com
model.load_state_dict(torch.load('checkpoint.pth', map_location='cpu'))
# только предсказание, без обучения
model.eval() 


results = {0: ['котофей', 'це кiт', 'очевидно, это кошка', 'какой славный котик'],
           1: ['что тут у нас, собака', 'собаня', 'судя по всему, здесь пёс', 'собака']}

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'Привет! я умею отличать собак от кошек. Пришли мне фото, и я пойму, кошка там или же собака')


@bot.message_handler(content_types=['photo'])
def photo(message):
    img_id = message.photo[-1].file_id
    img_info = bot.get_file(img_id)
    path = img_info.file_path
    downloaded_file = bot.download_file(path)
    with open('image.jpg', 'wb') as new_file:
        new_file.write(downloaded_file)
    result = torch.nn.functional.softmax(model(image_loader('image.jpg')).detach()).numpy()
    res = (result.max(), np.argmax(result))
    # порог для предсказания
    if res[0] > 0.9:
        bot.reply_to(message, choice(results[res[1]]))
    else:
        bot.reply_to(message, 'Извини, я не вижу здесь собак и кошек. Попробуй отправить мне фото получше!')



# непрерывное ожидание запросов ботом
bot.polling(none_stop=True, interval=0)
