class EDSSCalculator:
    def __init__(self):
        self.scores = {
            'mobility': 0,
            'sensory': 0,
            'visual': 0,
            'bladder_bowel': 0,
            'cognitive': 0,
            'motor': 0,
            'cerebellar': 0,
            'speech': 0,
            'mental_state': 0,
            'fatigue': 0
        }
    
    def assess_mobility(self, distance):
        if distance >= 1000:
            self.scores['mobility'] = 1
        elif 500 <= distance < 1000:
            self.scores['mobility'] = 2
        elif 200 <= distance < 500:
            self.scores['mobility'] = 3
        else:
            self.scores['mobility'] = 4

    def assess_sensory(self, sensory_type):
        self.scores['sensory'] = {
            'normal': 1,
            'mild': 2,
            'moderate': 3,
            'severe': 4,
        }.get(sensory_type, 4)

    def assess_visual(self, acuity_left, acuity_right):
        if acuity_left >= 1.0 and acuity_right >= 1.0:
            self.scores['visual'] = 1
        elif acuity_left >= 0.7 or acuity_right >= 0.7:
            self.scores['visual'] = 2
        else:
            self.scores['visual'] = 3

    def assess_bladder_bowel(self, function):
        if function == 'normal':
            self.scores['bladder_bowel'] = 1
        elif function == 'mild_incontinence':
            self.scores['bladder_bowel'] = 2
        else:
            self.scores['bladder_bowel'] = 3

    def assess_cognitive(self, level):
        self.scores['cognitive'] = {
            'normal': 1,
            'mild': 2,
            'moderate': 3,
            'severe': 4,
        }.get(level, 4)

    def assess_motor(self, strength_level):
        self.scores['motor'] = {
            'normal': 1,
            'mild': 2,
            'moderate': 3,
            'severe': 4,
        }.get(strength_level, 4)

    def assess_cerebellar(self, coordination_level):
        self.scores['cerebellar'] = {
            'normal': 1,
            'mild': 2,
            'severe': 3,
        }.get(coordination_level, 3)

    def assess_speech(self, speech_condition):
        self.scores['speech'] = {
            'normal': 1,
            'mild': 2,
            'moderate': 3,
            'severe': 4,
        }.get(speech_condition, 4)

    def assess_mental_state(self, mental_condition):
        self.scores['mental_state'] = {
            'normal': 1,
            'mild': 2,
            'moderate': 3,
            'severe': 4,
        }.get(mental_condition, 4)

    def assess_fatigue(self, fatigue_level):
        self.scores['fatigue'] = {
            'none': 1,
            'mild': 2,
            'moderate': 3,
            'severe': 4,
        }.get(fatigue_level, 4)

    def calculate_edss(self):
        total_score = sum(self.scores.values())
        total_score = min(total_score, 30)  # Максимум 30, соответствие докладу EDSS
        return total_score