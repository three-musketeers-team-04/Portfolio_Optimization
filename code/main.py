from user_interface import UI
from backend_app import process

app = UI(process_fn=process)
app.mainloop()