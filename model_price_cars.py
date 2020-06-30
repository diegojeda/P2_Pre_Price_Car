# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 00:27:39 2020

@author: Jorge
"""

#!/usr/bin/python

import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.externals import joblib
import xgboost as xgb
from xgboost.sklearn import XGBRegressor

import sys
import os

def predict_proba(Year,Mileage,State,Make,Model):
    
    Make_df=pd.DataFrame({'Make':['Acura', 'Audi', 'BMW', 'Bentley', 'Buick', 'Cadillac',
       'Chevrolet', 'Chrysler', 'Dodge', 'FIAT', 'Ford', 'Freightliner',
       'GMC', 'Honda', 'Hyundai', 'INFINITI', 'Jaguar', 'Jeep', 'Kia',
       'Land', 'Lexus', 'Lincoln', 'MINI', 'Mazda', 'Mercedes-Benz',
       'Mercury', 'Mitsubishi', 'Nissan', 'Pontiac', 'Porsche', 'Ram',
       'Scion', 'Subaru', 'Suzuki', 'Tesla', 'Toyota', 'Volkswagen',
       'Volvo'], 'Make_N':[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
       34, 35, 36, 37]})
    
    Model_df=pd.DataFrame({'Model':['MDX4WD', 'MDXAWD', 'RDXAWD', 'RDXFWD', 'TL4dr', 'TLAutomatic',
       'TSXAutomatic', 'A34dr', 'A44dr', 'A64dr', 'A8', 'Q5quattro',
       'Q7quattro', 'S44dr', 'TT2dr', '1', '3', '5', '6', '7',
       'X1xDrive28i', 'X3AWD', 'X3xDrive28i', 'X5AWD', 'X5xDrive35i',
       'Continental', 'EnclaveConvenience', 'EnclaveLeather',
       'EnclavePremium', 'LaCrosse4dr', 'LaCrosseAWD', 'LaCrosseFWD',
       'Lucerne4dr', 'Regal4dr', 'RegalGS', 'RegalPremium', 'RegalTurbo',
       'CTS', 'CTS-V', 'CTS4dr', 'DTS4dr', 'Escalade', 'Escalade2WD',
       'Escalade4dr', 'EscaladeAWD', 'SRXLuxury', 'STS4dr',
       'Avalanche2WD', 'Avalanche4WD', 'Camaro2dr', 'CamaroConvertible',
       'CamaroCoupe', 'Cobalt2dr', 'Cobalt4dr', 'Colorado2WD',
       'Colorado4WD', 'ColoradoCrew', 'ColoradoExtended', 'Corvette2dr',
       'CorvetteConvertible', 'CorvetteCoupe', 'CruzeLT', 'CruzeSedan',
       'EquinoxAWD', 'EquinoxFWD', 'Express', 'Impala4dr', 'ImpalaLS',
       'ImpalaLT', 'Malibu', 'Malibu1LT', 'Malibu4dr', 'MalibuLS',
       'MalibuLT', 'Monte', 'New', 'Silverado', 'SonicHatch',
       'SonicSedan', 'Suburban2WD', 'Suburban4WD', 'Suburban4dr',
       'Tahoe2WD', 'Tahoe4WD', 'Tahoe4dr', 'TahoeLS', 'TahoeLT',
       'TraverseAWD', 'TraverseFWD', '200LX', '200Limited', '200S',
       '200Touring', '300300C', '300300S', '3004dr', '300Base',
       '300Limited', '300Touring', 'PT', 'PacificaLimited',
       'PacificaTouring', 'Town', 'Caliber4dr', 'Challenger2dr',
       'ChallengerR/T', 'Charger4dr', 'ChargerSE', 'ChargerSXT',
       'Dakota2WD', 'Dakota4WD', 'Durango2WD', 'Durango4dr', 'DurangoAWD',
       'DurangoSXT', 'Grand', 'JourneyAWD', 'JourneyFWD', 'JourneySXT',
       'Ram', 'Sprinter', '500Pop', 'Econoline', 'EdgeLimited', 'EdgeSE',
       'EdgeSEL', 'EdgeSport', 'Escape4WD', 'Escape4dr', 'EscapeFWD',
       'EscapeLImited', 'EscapeLimited', 'EscapeS', 'EscapeSE',
       'EscapeXLT', 'Excursion137"', 'Expedition', 'Expedition2WD',
       'Expedition4WD', 'ExpeditionLimited', 'ExpeditionXLT', 'Explorer',
       'Explorer4WD', 'Explorer4dr', 'ExplorerBase', 'ExplorerEddie',
       'ExplorerFWD', 'ExplorerLimited', 'ExplorerXLT', 'F-1502WD',
       'F-1504WD', 'F-150FX2', 'F-150FX4', 'F-150King', 'F-150Lariat',
       'F-150Limited', 'F-150Platinum', 'F-150STX', 'F-150SuperCrew',
       'F-150XL', 'F-150XLT', 'F-250King', 'F-250Lariat', 'F-250XL',
       'F-250XLT', 'F-350King', 'F-350Lariat', 'F-350XL', 'F-350XLT',
       'FiestaS', 'FiestaSE', 'FlexLimited', 'FlexSE', 'FlexSEL',
       'Focus4dr', 'Focus5dr', 'FocusS', 'FocusSE', 'FocusSEL', 'FocusST',
       'FocusTitanium', 'Fusion4dr', 'FusionHybrid', 'FusionS',
       'FusionSE', 'FusionSEL', 'Mustang2dr', 'MustangBase',
       'MustangDeluxe', 'MustangGT', 'MustangPremium', 'MustangShelby',
       'Ranger2WD', 'Ranger4WD', 'RangerSuperCab', 'Super', 'Taurus4dr',
       'TaurusLimited', 'TaurusSE', 'TaurusSEL', 'TaurusSHO', 'Transit',
       'AcadiaAWD', 'AcadiaFWD', 'Canyon2WD', 'Canyon4WD', 'CanyonCrew',
       'CanyonExtended', 'Savana', 'Sierra', 'TerrainAWD', 'TerrainFWD',
       'Yukon', 'Yukon2WD', 'Yukon4WD', 'Yukon4dr', 'Accord', 'AccordEX',
       'AccordEX-L', 'AccordLX', 'AccordLX-S', 'AccordSE', 'CR-V2WD',
       'CR-V4WD', 'CR-VEX', 'CR-VEX-L', 'CR-VLX', 'CR-VSE', 'CR-ZEX',
       'Civic', 'CivicEX', 'CivicEX-L', 'CivicLX', 'CivicSi',
       'Element2WD', 'Element4WD', 'FitSport', 'OdysseyEX', 'OdysseyEX-L',
       'OdysseyLX', 'OdysseyTouring', 'Pilot2WD', 'Pilot4WD', 'PilotEX',
       'PilotEX-L', 'PilotLX', 'PilotSE', 'PilotTouring', 'RidgelineRTL',
       'RidgelineSport', 'S2000Manual', 'Accent4dr', 'Azera4dr',
       'Elantra', 'Elantra4dr', 'ElantraLimited', 'Genesis', 'Santa',
       'Sonata4dr', 'SonataLimited', 'SonataSE', 'TucsonAWD', 'TucsonFWD',
       'TucsonLimited', 'VeracruzAWD', 'VeracruzFWD', 'FX35AWD', 'G35',
       'G37', 'QX562WD', 'QX564WD', 'XF4dr', 'XJ4dr', 'XK2dr',
       'CherokeeLimited', 'CherokeeSport', 'Compass4WD',
       'CompassLatitude', 'CompassLimited', 'CompassSport', 'Liberty4WD',
       'LibertyLimited', 'LibertySport', 'Patriot4WD', 'PatriotLatitude',
       'PatriotLimited', 'PatriotSport', 'Wrangler', 'Wrangler2dr',
       'Wrangler4WD', 'WranglerRubicon', 'WranglerSahara',
       'WranglerSport', 'WranglerX', 'Forte', 'ForteEX', 'ForteLX',
       'ForteSX', 'Optima4dr', 'OptimaEX', 'OptimaLX', 'OptimaSX',
       'RioLX', 'Sedona4dr', 'SedonaEX', 'SedonaLX', 'Sorento2WD',
       'SorentoEX', 'SorentoLX', 'SorentoSX', 'Soul+', 'SoulBase',
       'Sportage2WD', 'SportageAWD', 'SportageEX', 'SportageLX',
       'SportageSX', 'Rover', 'CT', 'CTCT', 'ES', 'ESES', 'GS', 'GSGS',
       'GX', 'GXGX', 'IS', 'ISIS', 'LS', 'LSLS', 'LX', 'LXLX', 'RX',
       'RXRX', 'SC', 'MKXAWD', 'MKXFWD', 'MKZ4dr', 'Navigator',
       'Navigator2WD', 'Navigator4WD', 'Navigator4dr', 'Cooper',
       'CX-7FWD', 'CX-9AWD', 'CX-9FWD', 'CX-9Grand', 'CX-9Touring', 'MX5',
       'Mazda34dr', 'Mazda35dr', 'Mazda64dr', 'RX-84dr', 'C-Class4dr',
       'C-ClassC', 'C-ClassC300', 'C-ClassC350', 'E-ClassE',
       'E-ClassE320', 'E-ClassE350', 'M-ClassML350', 'SL-ClassSL500',
       'SLK-ClassSLK350', 'Milan4dr', 'Eclipse3dr', 'Galant4dr',
       'Lancer4dr', 'Outlander', 'Outlander2WD', 'Outlander4WD',
       '350Z2dr', 'Altima4dr', 'Armada2WD', 'Armada4WD', 'Frontier',
       'Frontier2WD', 'Frontier4WD', 'Maxima4dr', 'Murano2WD',
       'MuranoAWD', 'MuranoS', 'Pathfinder2WD', 'Pathfinder4WD',
       'PathfinderS', 'PathfinderSE', 'Quest4dr', 'RogueFWD', 'Sentra4dr',
       'Titan', 'Titan2WD', 'Titan4WD', 'Versa4dr', 'Versa5dr',
       'Xterra2WD', 'Xterra4WD', 'Xterra4dr', 'G64dr', 'Vibe4dr', '911',
       '9112dr', 'Boxster2dr', 'CayenneAWD', 'Cayman2dr', '15002WD',
       '15004WD', '1500Laramie', '1500Tradesman', '25002WD', '25004WD',
       '35004WD', 'tC2dr', 'xB5dr', 'xD5dr', 'Forester2.5X',
       'Forester4dr', 'Impreza', 'Impreza2.0i', 'ImprezaSport', 'Legacy',
       'Legacy2.5i', 'Legacy3.6R', 'Outback2.5i', 'Outback3.6R',
       'WRXBase', 'WRXLimited', 'WRXPremium', 'WRXSTI', 'Model',
       '4Runner2WD', '4Runner4WD', '4Runner4dr', '4RunnerLimited',
       '4RunnerRWD', '4RunnerSR5', '4RunnerTrail', 'Avalon4dr',
       'AvalonLimited', 'AvalonTouring', 'AvalonXLE', 'Camry', 'Camry4dr',
       'CamryBase', 'CamryL', 'CamryLE', 'CamrySE', 'CamryXLE',
       'Corolla4dr', 'CorollaL', 'CorollaLE', 'CorollaS', 'FJ',
       'Highlander', 'Highlander4WD', 'Highlander4dr', 'HighlanderBase',
       'HighlanderFWD', 'HighlanderLimited', 'HighlanderSE', 'Land',
       'Matrix5dr', 'Prius', 'Prius5dr', 'PriusBase', 'PriusFive',
       'PriusFour', 'PriusOne', 'PriusThree', 'PriusTwo', 'RAV4',
       'RAV44WD', 'RAV44dr', 'RAV4Base', 'RAV4FWD', 'RAV4LE',
       'RAV4Limited', 'RAV4Sport', 'RAV4XLE', 'Sequoia4WD', 'Sequoia4dr',
       'SequoiaLimited', 'SequoiaPlatinum', 'SequoiaSR5', 'Sienna5dr',
       'SiennaLE', 'SiennaLimited', 'SiennaSE', 'SiennaXLE', 'Tacoma2WD',
       'Tacoma4WD', 'TacomaBase', 'TacomaPreRunner', 'Tundra',
       'Tundra2WD', 'Tundra4WD', 'TundraBase', 'TundraLimited',
       'TundraSR5', 'Yaris', 'Yaris4dr', 'YarisBase', 'YarisLE', 'CC4dr',
       'Eos2dr', 'GLI4dr', 'GTI2dr', 'GTI4dr', 'Golf', 'Jetta', 'Passat',
       'Passat4dr', 'Tiguan2WD', 'TiguanS', 'TiguanSE', 'TiguanSEL',
       'Touareg4dr', 'C702dr', 'S60T5', 'S804dr', 'XC60AWD', 'XC60FWD',
       'XC60T6', 'XC704dr', 'XC90AWD', 'XC90FWD', 'XC90T6'],
        'Model_N':[  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
        13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
        26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
        39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
        52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
        65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
        78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
        91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
       104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
       117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
       130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
       143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
       156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168,
       169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
       182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
       195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
       208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
       221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233,
       234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246,
       247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259,
       260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272,
       273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285,
       286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298,
       299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311,
       312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324,
       325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337,
       338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350,
       351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363,
       364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376,
       377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389,
       390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402,
       403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415,
       416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428,
       429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441,
       442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454,
       455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467,
       468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480,
       481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493,
       494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506,
       507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519,
       520, 521, 522, 523, 524]})
    
    State_df=pd.DataFrame({'State':[' DC', ' HI', ' VA', ' CT', ' OH', ' VT', ' MI', ' AZ', ' FL',
       ' IN', ' MD', ' NV', ' CA', ' IL', ' NJ', ' KS', ' DE', ' KY',
       ' GA', ' PA', ' SC', ' NH', ' CO', ' ND', ' MO', ' MN', ' TN',
       ' WI', ' NC', ' MA', ' NY', ' OR', ' ID', ' WA', ' AK', ' RI',
       ' IA', ' OK', ' UT', ' AL', ' ME', ' NE', ' TX', ' LA', ' NM',
       ' WV', ' AR', ' MS', ' SD', ' MT', ' WY'],'State_N':[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]})

    clf = joblib.load(os.getcwd() + '/final_model.pkl') 

    car = pd.DataFrame({'Year':[Year],
                    'Mileage':[Mileage],
                    'Make':[Make],
                    'Model':[Model],
                    'State':[State]})
    car_1= pd.merge(car,
                 Make_df,
                 on='Make')
    car_1= pd.merge(car_1,
                 Model_df,
                 on='Model')
    car_1= pd.merge(car_1,
                 State_df,
                 on='State')
    car_1=car_1.drop(['State','Make','Model'],axis=1)

    car_1.rename(columns={'State_N':'State',
                      'Model_N':'Model',
                      'Make_N':'Make'}, 
                 inplace=True)
    
    # Make prediction
    p1 = clf.predict(car_1)[0]

    return p1


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add an URL')
        
    else:

        url = sys.argv[1]

        p1 = predict_proba(url)
        
        print(url)
        print('Probability of Phishing: ', p1)
        