
## Virtual Environment Setup
When you clone the repository, you will need to create a virtual environment and install the dependencies.

1. **Create a virtual environment**:
   ```sh
   py -3.11 -m venv myenv
   ```

2. **Activate the virtual environment**:
   ```sh
   myenv\Scripts\activate
   ```

3. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

- Activate the virtual environment before running any scripts or installing new packages and deactivate it when you are done working on the project.
- If you encounter the ```ImportError: DLL load failed while importing _framework_bindings``` error, Install Visual C++ Redistributable from [here](https://aka.ms/vs/17/release/vc_redist.x64.exe).