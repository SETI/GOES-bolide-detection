# This module is used to perofmr the validation of each candidate. 
# Nominally, this is to validate bolides, however it could also validate other phenomena in the data

import os
import time

from PIL import Image, ImageFont, ImageDraw 

import io_utilities as ioUtil
import bolide_dispositions as bDisp

TRIAGE_ASSESSMENT_OPTIONS = ('N/A', 'rejected', 'accepted')
VALIDATION_ASSESSMENT_OPTIONS = ('N/A', 'rejected', 'candidate', 'accepted')

# Bolide human confidence 
# neo-bolides website uses the confidence of {'low', 'medium', 'high'}
# Define confidence as low = 0.5, medium = 0.75, high = 0.95
# Auto-publish bolides have a confidence of 'auto', give thjese a belief score of 0.9
HUMAN_ASSESSMENT_OPTIONS = ('N/A', 'rejected', 'low', 'medium', 'high')
bolideWebsiteBeliefSwitcher = {
    'rejected': 0.25,
    'low': 0.5,
    'medium': 0.75,
    'high': 0.95,
    'auto': 0.9,
    'unknown': -1
    }


class TriageAssessment():
    """ This contains the triage assessment information
    If the threshold is not passed then the assessment cannot occur.
    """

    def __init__(self, threshold):

        self.threshold = threshold

        self._score = None
        self.method = None

        self.candidacy_forced = False

    @property           
    def score(self):
        return self._score
     
    @score.setter   
    def score(self, score):

        score = float(score)

        assert score >= 0.0 and score <= 1.0, 'Score must be in the range [0.0,1.0]'

        self._score = score

    @property           
    def assessment(self):

        if self.candidacy_forced:
            return 'accepted'
        elif self.score is None or self.threshold is None:
            return 'N/A'
        elif self.score < self.threshold:
            return 'rejected'
        elif self.score >= self.threshold:
            return 'accepted'
        else:
            raise Exception('Error determining assessment')

class ValidationAssessment():
    """ This contains the validation assessment information
    """

    def __init__(self, low_threshold, high_threshold):

        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        self._score = None
        self.method = None

        self.candidacy_forced = False

    @property           
    def score(self):
        return self._score
     
    @score.setter   
    def score(self, score):

        assert score is None or (score >= 0.0 and score <= 1.0), 'Score must be in the range [0.0,1.0] or None'

        if score is None:
            self._score = None
        else:
            score = float(score)

        self._score = score
     
    @property           
    def assessment(self):

        # We want to force the candidacy but only if the score is nor >= the high threshold (which would make it "accepted')
        if self.candidacy_forced and (self.score is None or self.score < self.high_threshold):
            return 'candidate'
        elif self.score is None or self.low_threshold is None or self.high_threshold is None:
            return 'N/A'
        elif self.score < self.low_threshold:
            return 'rejected'
        elif self.score < self.high_threshold:
            return 'candidate'
        elif self.score >= self.high_threshold:
            return 'accepted'
        else:
            raise Exception('Error determining assessment')
         
class HumanAssessment():
    """ This contains the human assessment information

    For a human assessment, instead of giving a numerical score, we give a string: {'rejected', 'low', 'medium', 'high'}
    We then create a score value based on that, as given by 
    """

    def __init__(self, assessment=None, source=None):

        self.source = source
        self.assessment = assessment

    @property           
    def assessment(self):
        return self._assessment
     
    @assessment.setter   
    def assessment(self, assessment):

        if assessment is None:
            assessment = 'N/A'

        assert assessment in HUMAN_ASSESSMENT_OPTIONS, 'Assessment is not valid'

        self._assessment = assessment
         
    @property           
    def score(self):

        score = bolideWebsiteBeliefSwitcher.get(self.assessment, bolideWebsiteBeliefSwitcher['unknown'])
        
        return score
     

class BolideAssessment():
    """ This class is used to contain the assesment of the "bolideness" of each detection candidate

    """

    def __init__(self, triage_threshold=None, validation_low_threshold=None, validation_high_threshold=None):
        """ Initialize
        Initializing the threshold is optional. If thrsholds ar enot given, then they must be set later before the
        assessment can be performed.

        """

        self.triage = TriageAssessment(triage_threshold)

        self.validation = ValidationAssessment(validation_low_threshold, validation_high_threshold)

        self.human = HumanAssessment()
        
    def __repr__(self): 

        
        printDict = {}
        if self.triage.score is None:
            printDict['Triage not available'] = None
        else:
            printDict['Triage threshold'] = self.triage.threshold
            printDict['Triage Score'] = self.triage.score
            printDict['Triage assessment'] = self.triage.assessment
            printDict['Triage method'] = self.triage.method
            printDict['Triage candidacy forced'] = self.triage.candidacy_forced

        if self.validation.score is None:
            printDict['Validation not available'] = None
        else:
            printDict['Validation low threshold'] = self.validation.low_threshold
            printDict['Validation high threshold'] = self.validation.high_threshold
            printDict['Validation Score'] = self.validation.score
            printDict['Validation assessment'] = self.validation.assessment
            printDict['Validation method'] = self.validation.method
            printDict['Validation candidacy forced'] = self.validation.candidacy_forced

        if self.human.assessment is None:
            printDict['Human not available'] = None
        else:
            printDict['Human assessment'] = self.human.assessment
            printDict['Human score'] = self.human.score
            printDict['Human source'] = self.human.source

        return ioUtil.print_dictionary(printDict)


def evaluate_bolide_candidates(validation_input_config, bolideDetectionList):
    """ Evaluates the trained validation model on the bolide candidates

    Parameters
    ----------
    validation_input_config : validation_io.validation_input_config
    bolideDetectionList : list of BolideDetection

    Returns
    -------
    bolideDetectionList : List of BolideDetection
        The bolide detection candidates with the validation score added
        If validation cannot be performed then bolideDetectionList[:].assessment.validation.score = None

    """
    
    # We need to place this import statement here so that we do not have a circular import in bolide_cnn.py
    from bolide_cnn import CnnEvaluator, clear_tmp_directory

    startTime = time.time()


    if len(bolideDetectionList) == 0:
        return bolideDetectionList

    if validation_input_config.validation_model_path is None:
        for detection in bolideDetectionList:
            detection.assessment.validation.score = None
        return bolideDetectionList

    # Convert to bolide dispositions
    bolideDispositionProfileList = []
    for detection in bolideDetectionList:
        bolideDispositionProfileList.append(bDisp.BolideDispositionProfile(detection.ID, 
            detectionFlag=True, 
            bolideDetection=detection, 
            machineOpinions=[bDisp.MachineOpinion(bolideBelief=detection.assessment.triage.score,
                                method=detection.assessment.triage.method, comments=detection.howFound)], 
            features=detection.features,
            stereoFeatures=detection.stereoFeatures, 
            cutoutFeatures=detection.cutoutFeatures ))
    bDispObj = bDisp.BolideDispositions.from_bolideDispositionProfileList(bolideDispositionProfileList, useRamDisk=False)

    # Create a CnnEvaluator object and load the model
    cnnEvaluator = CnnEvaluator(validation_input_config.validation_model_path, 
            bDispObj,
            image_cache_path=validation_input_config.validation_image_cache_path,
            gpu_index=validation_input_config.gpu_index, 
            force_cpu=False, verbosity=False)

    # If CnnEvaluator creation was not successful then we cannot proceed
    if not cnnEvaluator.init_success:
        return bolideDetectionList

    # Evaluate model!
    _, _, _ = cnnEvaluator.evaluate('all')
    # Populate the validation scores
    bolideDetectionList = cnnEvaluator.populate_validation_scores_in_bolideDetectionList(validation_input_config, bolideDetectionList)
    if validation_input_config.delete_image_pickle_files: 
        cnnEvaluator.delete_image_pickle_files()
    
    # Clear out image cache
    clear_tmp_directory(paths_to_clear=[validation_input_config.validation_image_cache_path], verbosity=False)
    
    endTime = time.time()
    totalTime = endTime - startTime
    print("Bolide validation time: {:.2f} seconds, {:.2f} minutes".format(totalTime, totalTime / 60))

    return bolideDetectionList

def add_validation_assessment_to_detection_figures(detection_input_config, validation_input_config, bolideDetectionList, detectionFigurePath):
    """ Adds the bolide validation assessment to the detection figures.

    Parameters
    ----------
    detection_input_config : bolide_io.input_config
    validation_input_config : validation_io.validation_input_config
    bolideDetectionList : list of BolideDetection
    detectionFigurePath : str
        Path to all the detection, cutout and post-processing figures in bolideDetectionList
        This is the full path to the individual files, not a top level path with subdirectories

    Returns
    -------
    Nothing, just modified figures
    """

    # If not generating figures then do nothing
    if not detection_input_config.generatePlots:
        return

    # Define color scheme
    color_scheme = {
            'N/A': (194, 106, 119),
            'rejected': (148, 203, 236),
            'candidate': (46, 37, 133),
            'accepted': (51, 117, 56)
            }

    for detection in bolideDetectionList:
        # Get the detection figure
        detectionFileThisID = os.path.join(detectionFigurePath, detection.figureFilename)
        assert os.path.exists(detectionFileThisID), 'Found no detection figure for detection ID: {}, This should not be.'.format(detection.ID)

        # Determine assessment for this candidate
        assert detection.assessment.triage.assessment in color_scheme, 'Unkown triage assessment'
        triage_color = color_scheme[detection.assessment.triage.assessment]

        assert detection.assessment.validation.assessment in color_scheme, 'Unkown validation assessment'
        validation_color = color_scheme[detection.assessment.validation.assessment]

        # Write the assessment
        img = Image.open(detectionFileThisID)
        draw = ImageDraw.Draw(img)
       #font = ImageFont.truetype("sans-serif.ttf", 10)
        font_size = 23

        # Triage
        if detection.assessment.triage.assessment == 'N/A':
            draw.text((1190, 35),"Triage Score: 'N/A'", triage_color, font_size=font_size)
        else:
            draw.text((1200, 2),"Triage: ",(0,0,0), font_size=font_size)
            draw.text((1300, 2),"{}".format(detection.assessment.triage.assessment), triage_color, font_size=font_size)

        # Validation
        if detection.assessment.validation.assessment == 'N/A':
            draw.text((1190, 35),"Validation Score: 'N/A'", validation_color, font_size=font_size)
        else:
            draw.text((1190, 35),"Validation Score: {:.4f}".format(detection.assessment.validation.score),(0,0,0), font_size=font_size)

        draw.text((1190, 65),"Validation:",(0,0,0), font_size=font_size)
        draw.text((1320, 65),"{}".format(detection.assessment.validation.assessment),validation_color, font_size=font_size)
        img.save(detectionFileThisID)

    return
