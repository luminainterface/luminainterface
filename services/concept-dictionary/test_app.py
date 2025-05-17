from main import app
import uvicorn
 
if __name__ == "__main__":
    print("Available routes:")
    for route in app.routes:
        print(f"{route.methods} {route.path}")
    uvicorn.run(app, host="0.0.0.0", port=8000) 