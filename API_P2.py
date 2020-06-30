from flask import Flask
from flask_restx import Api, Resource, fields
from sklearn.externals import joblib
from model_price_cars import predict_proba

app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Price Car Prediction - Group 9',
    description='Esta aplicación permite estimar el precio de venta de un automovil en EEUU a partir del año de fabricación, las millas recorridas, el estado del que proviene, la marca y el modelo.')

ns = api.namespace('predict', 
     description='Price prediction')
   
parser = api.parser()


parser.add_argument(
    'Year', 
    type=float, 
    required=True, 
    help='Año de fabricación del vehículo', 
    location='args')

parser.add_argument(
    'Mileage', 
    type=float, 
    required=True, 
    help='Recorrido del vehículo', 
    location='args')

parser.add_argument(
    'State', 
    type=str, 
    required=True, 
    help='Estado en formato dos letras, precedido por un espacio', 
    location='args')

parser.add_argument(
    'Make', 
    type=str, 
    required=True, 
    help='Marca del automóvil', 
    location='args')

parser.add_argument(
    'Model', 
    type=str, 
    required=True, 
    help='Línea de la marca del automóvil', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})


@ns.route('/')
class PhishingApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        Year = args['Year']
        Mileage = args['Mileage']
        State = args['State']
        Make = args['Make']
        Model = args['Model']
        
        print(Year)
        print(Mileage)
        print(State)
        print(Make)
        print(Model)
        
        
        return {
         "result": predict_proba(Year,Mileage,State,Make,Model)
        }, 200

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8000)

        