import string
import math
from data import data


class Model:
    def __init__(self) -> None:
        self.tf_idf = {}

    def convert_to_lists(self,  sentences: list) -> list:
        ret = []
        for sentence in sentences:
            sentence = sentence.translate(
                str.maketrans('', '', string.punctuation)).lower()
            ret.append(sentence.split())



            
        return ret

    def get_avg_tf(self, sentences: list) -> dict:
        freq = {}
        for str_arr in sentences:
            for word in str_arr:
                if word not in freq:
                    freq[word] = 0
                freq[word] += (1 / len(str_arr)) / len(sentences)
        return freq

    def get_idf(self, sentences: list) -> dict:
        freq = {}
        for str_arr in sentences:
            for word in set(str_arr):
                if word not in freq:
                    freq[word] = 0
                freq[word] += 1
        for key in freq:
            freq[key] = math.log(len(sentences) / freq[key], 10)
        return freq


    def train(self, data):
        for subject in data:
            str_lists = self.convert_to_lists(data[subject])
            tf = self.get_avg_tf(str_lists)
            idf = self.get_idf(str_lists)
            self.tf_idf[subject] = {}
            for key in tf.keys() & idf.keys():
                self.tf_idf[subject][key] = round(tf[key] * idf[key], 3)

    def check(self, sentence):
        str_lists = self.convert_to_lists([sentence])
        tf = self.get_avg_tf(str_lists)
        max_sub = ""
        max_val = 0
        for sub in self.tf_idf:
            curr_val = 0
            for key in self.tf_idf[sub].keys() & tf.keys():
                curr_val += self.tf_idf[sub][key] * tf[key]
            if curr_val > max_val:
                max_val = curr_val
                max_sub = sub
        return max_sub


model = Model()
model.train(data)
print(model.tf_idf)

# Additional sample sentences to check
additional_tech_sentences = [
    "Artificial neural networks are being used to improve natural language processing tasks.",
    "The development of self-driving cars relies heavily on sensor technology and machine learning algorithms.",
    "Cloud computing services offer scalability and cost-effectiveness for businesses of all sizes.",
    "Advancements in biotechnology are driving innovation in the healthcare industry.",
    "Renewable energy technologies such as solar panels and wind turbines are becoming more efficient and affordable.",
    "Virtual reality applications are expanding beyond entertainment into areas like education and therapy.",
    "The Internet of Things has the potential to revolutionize healthcare by enabling remote patient monitoring.",
    "Blockchain technology is being explored for its potential applications in supply chain management and digital identity verification.",
    "The use of drones in agriculture can optimize crop management and yield predictions.",
    "Cybersecurity threats continue to evolve, prompting the need for constant vigilance and advanced defense mechanisms."
]

additional_health_sentences = [
    "Adequate sleep is important for overall health and cognitive function.",
    "Regular medical screenings can detect conditions such as high blood pressure and diabetes early.",
    "Mental health awareness campaigns aim to reduce stigma and increase access to support services.",
    "The Mediterranean diet is associated with numerous health benefits, including reduced risk of heart disease.",
    "Physical activity guidelines recommend at least 150 minutes of moderate exercise per week for adults.",
    "Mindfulness meditation practices can reduce stress and improve emotional well-being.",
    "Regular handwashing is one of the most effective ways to prevent the spread of infectious diseases.",
    "Proper nutrition during pregnancy is essential for fetal development and maternal health.",
    "Chronic stress can contribute to a variety of health problems, including digestive issues and cardiovascular disease.",
    "Social support networks play a crucial role in mental health recovery and resilience."
]

for sent in additional_health_sentences:
    print(model.check(sent))

for sent in additional_tech_sentences:
    print(model.check(sent))