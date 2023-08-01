# Do not edit if deploying to Banana Serverless
# This file is boilerplate for the http server, and follows a strict interface.

# Instead, edit the init() and inference() functions in app.py

from potassium import Potassium, Request, Response
import subprocess
import user_src 

# We do the model load-to-GPU step on server startup
# so the model object is available globally for reuse
app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    context = user_src.init()
    return context

# Inference POST handler at '/' is called for every http call from Banana
@app.handler('/')
def inference(context: dict, request: Request) -> Response:
    model_inputs =  request.json
    model = context.get('model')

    output = user_src.inference(model, model_inputs)

    return Response(
        json = {"outputs": output}, 
        status=200
    )

if __name__ == '__main__':
    app.serve()