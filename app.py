import os
import pandas as pd
from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    jsonify, session, send_file
)
# *** Import Session ***
from flask_session import Session
from werkzeug.utils import secure_filename
from sqlalchemy import create_engine, text, exc
import secrets
import io
import uuid
from datetime import datetime
import re

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
SAVED_SESSIONS_FOLDER = 'saved_sessions'
ALLOWED_EXTENSIONS = {'csv', 'tsv', 'xlsx'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SAVED_SESSIONS_FOLDER'] = SAVED_SESSIONS_FOLDER
# IMPORTANT: Change this to a strong, persistent secret key in production!
# Keep it outside version control.
app.config['SECRET_KEY'] = secrets.token_hex(24)
app.config['SESSION_TYPE'] = 'filesystem'  # Store sessions in files
app.config['SESSION_FILE_DIR'] = './.flask_session' # Directory to store session files
app.config['SESSION_PERMANENT'] = False # Session expires when browser closes
app.config['SESSION_USE_SIGNER'] = True # Encrypt session cookie

# --- Initialize Flask-Session ---
# IMPORTANT: Do this *after* setting app.config values
server_session = Session(app)

# --- Create session directory if it doesn't exist ---
if not os.path.exists(app.config['SESSION_FILE_DIR']):
    os.makedirs(app.config['SESSION_FILE_DIR'])


# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, SAVED_SESSIONS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# --- Helper Functions ---

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_dataframe_from_session():
    """Safely retrieve the current DataFrame from session."""
    df_json = session.get('current_df_json')
    if df_json:
        try:
            # Read JSON string back into DataFrame
            # orient='split' is often efficient for round-tripping
            return pd.read_json(io.StringIO(df_json), orient='split')
        except Exception as e:
            flash(f"Error loading data from session: {e}", "error")
            clear_session_data() # Clear corrupted data
            return None
    return None

def store_dataframe_in_session(df):
    """Store DataFrame in session as JSON string."""
    if df is None:
         session.pop('current_df_json', None)
         return
    try:
        # Convert DataFrame to JSON string using 'split' orientation
        # Handle potential date issues during serialization
        df_json = df.to_json(orient='split', date_format='iso', default_handler=str)
        session['current_df_json'] = df_json
    except Exception as e:
        flash(f"Error storing data in session: {e}. Data might be too large or contain unserializable types.", "error")
        # Decide how to handle: maybe clear session or keep old data?
        # session.pop('current_df_json', None) # Option: clear if storing fails

def add_to_undo(current_df_json):
    """Add current state to undo history (limited size)."""
    if not current_df_json: return
    undo_history = session.get('undo_history', [])
    undo_history.append(current_df_json)
    # Limit history size to prevent huge sessions
    max_history = 10
    session['undo_history'] = undo_history[-max_history:]
    session['redo_history'] = [] # Clear redo on new action

def get_undo_redo_status():
    """Return boolean status for undo/redo availability."""
    return {
        'undo_enabled': bool(session.get('undo_history')),
        'redo_enabled': bool(session.get('redo_history'))
    }

# Modified: render_table_html now optionally returns counts too
def render_table_html(df, max_rows=50):
     """Generates HTML for the table preview and returns dimensions."""
     if df is None or df.empty:
         table_html = "<p class='text-center text-muted p-4'>No data to display or data is empty.</p>"
         total_rows = 0
         total_columns = 0
     else:
         display_df = df.head(max_rows)
         table_html = display_df.to_html(
             classes='table table-striped table-bordered table-hover table-sm', # smaller table
             index=False,
             border=0,
             escape=True,
             na_rep='<i class="text-muted">NULL</i>'
         )
         total_rows = len(df) # Get total rows from the full DataFrame
         total_columns = len(df.columns) # Get total columns

     return table_html, total_rows, total_columns

def clear_session_data():
    """Clears all data related to the current cleaning session."""
    session.pop('current_df_json', None)
    session.pop('undo_history', None)
    session.pop('redo_history', None)
    session.pop('source_info', None)
    session.pop('saved_filename', None) # Also clear saved file link

# --- Routes ---

@app.route('/')
def index():
    """Homepage: Upload, Connect DB, or Open Saved File."""
    # List saved files (simple listing, could be enhanced)
    saved_files = []
    try:
        saved_files = [f for f in os.listdir(app.config['SAVED_SESSIONS_FOLDER']) if f.endswith('.pkl')]
    except FileNotFoundError:
        pass # Folder might not exist yet
    return render_template('index.html', saved_files=saved_files)

def load_data_into_session(df, source_info):
    """Helper to load df into session and redirect."""
    if df is not None:
        clear_session_data() # Start fresh
        store_dataframe_in_session(df)
        session['source_info'] = source_info
        return redirect(url_for('clean_data_interface'))
    else:
        flash('Could not load data.', 'error')
        return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload, loads data into session."""
    # (Similar upload logic as before, but use load_data_into_session)
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Save temporarily, process, then potentially remove
        # Or read directly from file stream if possible and safe
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            df = None
            file_ext = filename.rsplit('.', 1)[1].lower()
            if file_ext == 'csv':
                 # Try detecting separator, fall back to comma
                 try:
                     df = pd.read_csv(filepath, sep=None, engine='python', on_bad_lines='warn')
                     if df.shape[1] <= 1:
                          df = pd.read_csv(filepath, on_bad_lines='warn')
                 except Exception as read_err:
                      flash(f'Could not automatically determine CSV separator, trying comma. Error: {read_err}', 'warning')
                      df = pd.read_csv(filepath, on_bad_lines='warn')
            elif file_ext == 'tsv':
                df = pd.read_csv(filepath, sep='\t', on_bad_lines='warn')
            elif file_ext == 'xlsx':
                # Consider asking for sheet name if multiple sheets exist
                df = pd.read_excel(filepath)

            os.remove(filepath) # Clean up temporary file
            return load_data_into_session(df, f"Uploaded File: {filename}")

        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'error')
            if os.path.exists(filepath): # Clean up if error occurred after save
                 try:
                     os.remove(filepath)
                 except OSError: pass # Ignore error if removal fails
            return redirect(url_for('index'))
    else:
        flash('Invalid file type.', 'error')
        return redirect(url_for('index'))


@app.route('/database_form', methods=['GET'])
def database_form():
    """Shows the database connection form."""
    return render_template('database_form.html')

@app.route('/database_query', methods=['POST'])
def database_query():
    """Connects to DB, loads data into session."""
    # (Similar DB logic as before, but use load_data_into_session)
    db_type = request.form.get('db_type')
    db_host = request.form.get('db_host')
    db_port = request.form.get('db_port')
    db_name = request.form.get('db_name') # For SQLite, this is the file path
    db_user = request.form.get('db_user')
    db_password = request.form.get('db_password')
    query = request.form.get('query')

    if not query:
        flash('SQL Query cannot be empty.', 'error')
        return redirect(url_for('database_form'))

    connection_string = None
    engine = None
    source_info = f"Database: {db_type}"
    df = None

    try:
        # ... (Connection string logic same as before) ...
        if db_type == 'sqlite':
            connection_string = f"sqlite:///{db_name}"
            source_info = f"SQLite: {db_name}"
        elif db_type == 'postgresql':
            connection_string = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            source_info = f"PostgreSQL: {db_user}@{db_host}:{db_port}/{db_name}"
        elif db_type == 'mysql':
             connection_string = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
             source_info = f"MySQL: {db_user}@{db_host}:{db_port}/{db_name}"
        else: # Should have validation on form too
             flash('Unsupported database type.', 'error')
             return redirect(url_for('database_form'))

        engine = create_engine(connection_string)
        with engine.connect() as connection:
            df = pd.read_sql(text(query), connection)
        flash('Query successful.', 'success')

    except ImportError as e:
         flash(f"Database driver error: {e}. Make sure the required driver is installed.", "error")
         return redirect(url_for('database_form'))
    except exc.SQLAlchemyError as e:
        flash(f"Database connection or query error: {e}", "error")
        return redirect(url_for('database_form'))
    except Exception as e:
        flash(f"An unexpected error occurred: {str(e)}", "error")
        return redirect(url_for('database_form'))
    finally:
        if engine:
            engine.dispose()

    return load_data_into_session(df, source_info + f" (Query: {query[:50]}...)")


@app.route('/clean')
def clean_data_interface():
    """Displays the main data cleaning interface."""
    df = get_dataframe_from_session()
    if df is None:
        flash("No data loaded. Please upload a file or connect to a database.", "warning")
        return redirect(url_for('index'))

    table_html, total_rows, total_columns = render_table_html(df)
    columns = df.columns.tolist() if df is not None else []
    source_info = session.get('source_info', 'Unknown Source')
    saved_filename = session.get('saved_filename') # Get saved filename if exists

    return render_template(
        'clean_data.html',
        table_html=table_html,
        columns=columns,
        source_info=source_info,
        undo_redo_status=get_undo_redo_status(),
        saved_filename=saved_filename,
        total_rows=total_rows,
        total_columns=total_columns
    )


# --- AJAX Endpoints for Cleaning Operations ---

@app.route('/clean_operation', methods=['POST'])
def handle_clean_operation():
    """
    Generic handler for cleaning operations triggered by AJAX.
    Retrieves data from session, performs operation, updates session,
    and returns JSON response for the frontend.
    """
    # 1. Get the current DataFrame from session
    df = get_dataframe_from_session()
    if df is None:
        # Return error if no data is loaded in the session
        return jsonify({'error': 'No data loaded in session. Please load data first.'}), 400

    # 2. Get operation details from the JSON request body
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({'error': 'Invalid request format. JSON body expected.'}), 400
        operation = request_data.get('operation')
        params = request_data.get('params', {})
        if not operation:
             return jsonify({'error': 'Missing "operation" in request.'}), 400
    except Exception as e:
        app.logger.error(f"Error parsing request JSON: {e}")
        return jsonify({'error': f'Error parsing request: {str(e)}'}), 400

    # 3. Store current state for Undo BEFORE modifying the DataFrame
    current_df_json = session.get('current_df_json') # Get the JSON string directly
    # Ensure add_to_undo handles the case where current_df_json might be None initially,
    # though we check for df existence earlier.
    add_to_undo(current_df_json) # This helper should also clear the redo history

    # 4. Perform the requested cleaning operation
    try:
        original_shape = df.shape
        action_msg = "" # To store the success message

        # --- Apply the requested operation ---
        if operation == 'remove_duplicates':
            subset = params.get('subset') # Optional: list of columns, None means all
            # Ensure subset columns exist if provided (basic check)
            if subset and not all(col in df.columns for col in subset):
                 return jsonify({'error': 'One or more subset columns not found.'}), 400
            df_new = df.drop_duplicates(subset=subset, keep='first')
            removed_count = original_shape[0] - df_new.shape[0]
            action_msg = f"Removed {removed_count} duplicate row(s)."
            df = df_new # Assign the result back to df

        elif operation == 'remove_missing':
            how = params.get('how', 'any') # 'any' or 'all'
            subset = params.get('subset') # Optional: list of columns
            if how not in ['any', 'all']:
                 return jsonify({'error': 'Invalid value for "how". Must be "any" or "all".'}), 400
            if subset and not all(col in df.columns for col in subset):
                 return jsonify({'error': 'One or more subset columns not found.'}), 400
            df_new = df.dropna(axis=0, how=how, subset=subset) # axis=0 for rows
            removed_count = original_shape[0] - df_new.shape[0]
            action_msg = f"Removed {removed_count} row(s) containing missing values."
            df = df_new

        elif operation == 'fill_missing':
            fill_value_str = params.get('value', '') # Get fill value as string initially
            column = params.get('column') # Optional: specific column name

            if column: # Fill specific column
                 if column not in df.columns:
                      return jsonify({'error': f'Column "{column}" not found'}), 400

                 # Attempt to convert fill value to the column's native type if possible
                 fill_value = fill_value_str # Default to string
                 try:
                     col_dtype = df[column].dtype
                     if pd.api.types.is_numeric_dtype(col_dtype):
                          # Allow empty string to be treated as NaN for numeric types if desired
                          # Or force conversion and handle error
                          if fill_value_str == '':
                               fill_value = pd.NA # Use pandas NA for consistency
                          else:
                               fill_value = pd.to_numeric(fill_value_str)
                     elif pd.api.types.is_datetime64_any_dtype(col_dtype):
                          if fill_value_str == '':
                               fill_value = pd.NaT # Use pandas NaT for datetime
                          else:
                               fill_value = pd.to_datetime(fill_value_str) # Let pandas parse
                     elif pd.api.types.is_bool_dtype(col_dtype):
                          # Handle boolean conversion carefully
                          lowered_val = fill_value_str.lower()
                          if lowered_val in ['true', '1', 'yes']: fill_value = True
                          elif lowered_val in ['false', '0', 'no']: fill_value = False
                          elif lowered_val == '': fill_value = pd.NA # Or False? Decide behavior
                          else: raise ValueError("Invalid boolean value") # Raise error for ambiguity
                     # else: keep as string (already assigned)

                 except (ValueError, TypeError) as conv_err:
                     # If conversion fails, maybe warn user but proceed with string? Or return error?
                     # Option 1: Return error
                     # return jsonify({'error': f'Could not convert "{fill_value_str}" to type of column "{column}". Error: {conv_err}'}), 400
                     # Option 2: Warn and use string (might change column dtype)
                      action_msg += f" (Warning: Could not convert '{fill_value_str}' to column type, used as string)."
                      fill_value = fill_value_str # Ensure it's the string version

                 df[column] = df[column].fillna(fill_value)
                 action_msg = f"Filled missing values in column '{column}' with '{fill_value_str}'." + action_msg

            else: # Fill all columns
                 # Simple approach: fill all with the same string value.
                 # More advanced: could try type conversion per column, but complex.
                 df = df.fillna(fill_value_str) # Use the original string value
                 action_msg = f"Filled missing values in all columns with '{fill_value_str}'."

        elif operation == 'remove_spaces':
             column = params.get('column')
             if not column:
                  return jsonify({'error': 'Column parameter is required for removing spaces.'}), 400
             if column not in df.columns:
                  return jsonify({'error': f'Column "{column}" not found'}), 400

             # Only apply to object/string columns
             if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
                 # Ensure column is treated as string, then strip
                 df[column] = df[column].astype(str).str.strip()
                 action_msg = f"Removed leading/trailing spaces from column '{column}'."
             else:
                  action_msg = f"Operation skipped: Remove spaces only applicable to text columns (column '{column}' is not text)."
                  # Don't return error, just inform user maybe? Or return 400?
                  # return jsonify({'error': f'Remove spaces only applicable to text columns (column "{column}" is not text)'}), 400


        elif operation == 'fix_datetime':
             column = params.get('column')
             date_format = params.get('format') # Optional format string (e.g., '%Y-%m-%d')
             if not column:
                  return jsonify({'error': 'Column parameter is required for fixing dates.'}), 400
             if column not in df.columns:
                  return jsonify({'error': f'Column "{column}" not found'}), 400

             # errors='coerce' will turn unparseable dates into NaT (Not a Time/Null)
             original_nulls = df[column].isnull().sum()
             df[column] = pd.to_datetime(df[column], format=date_format, errors='coerce')
             new_nulls = df[column].isnull().sum()
             coerced_count = new_nulls - original_nulls

             action_msg = f"Converted column '{column}' to datetime type."
             if coerced_count > 0:
                 action_msg += f" {coerced_count} value(s) could not be parsed and were set to null."
        

        # === Check IDs ===
        elif operation == 'check_id_uniqueness':
            df_modified = False # This operation doesn't change the DataFrame
            column = params.get('column')
            if not column:
                 return jsonify({'error': 'Column parameter is required.'}), 400
            if column not in df.columns:
                 return jsonify({'error': f'Column "{column}" not found'}), 400

            if df[column].is_unique:
                action_msg = f"Column '{column}' contains unique values."
            else:
                duplicates = df[df[column].duplicated()][column].unique()
                dup_count = df[column].duplicated().sum()
                preview = duplicates[:5].tolist() # Show first 5 duplicates
                action_msg = f"Column '{column}' contains {dup_count} duplicate value(s). Examples: {preview}"

        elif operation == 'check_id_format':
            df_modified = False # This operation doesn't change the DataFrame
            column = params.get('column')
            pattern = params.get('pattern')
            if not column:
                 return jsonify({'error': 'Column parameter is required.'}), 400
            if column not in df.columns:
                 return jsonify({'error': f'Column "{column}" not found'}), 400
            if not pattern:
                 return jsonify({'error': 'Pattern parameter is required for format check.'}), 400

            try:
                # Attempt to compile regex to check validity early
                re.compile(pattern)
                # Use na=False to treat NaN as non-matching
                matches = df[column].astype(str).str.match(f'^{pattern}$', na=False) # Anchor pattern
                non_matching_count = (~matches).sum()
                if non_matching_count == 0:
                    action_msg = f"All values in column '{column}' match the pattern."
                else:
                    action_msg = f"Checked format for column '{column}'. {non_matching_count} value(s) did not match the pattern '{pattern}'."
            except re.error as e:
                 return jsonify({'error': f'Invalid regex pattern provided: {e}'}), 400
            except Exception as e:
                 return jsonify({'error': f'Error during format check: {e}'}), 500


        # === Correcting Numerical Outliers ===
        elif operation == 'remove_outliers_iqr':
            column = params.get('column')
            factor = params.get('factor', 1.5) # Default factor
            if not column:
                 return jsonify({'error': 'Column parameter is required.'}), 400
            if column not in df.columns:
                 return jsonify({'error': f'Column "{column}" not found'}), 400
            if not pd.api.types.is_numeric_dtype(df[column]):
                 return jsonify({'error': f'Column "{column}" must be numeric for IQR outlier removal.'}), 400
            try:
                 factor = float(factor)
                 if factor <= 0: raise ValueError("Factor must be positive")
            except (ValueError, TypeError):
                 return jsonify({'error': 'Invalid factor provided. Must be a positive number.'}), 400

            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR

            df_new = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].reset_index(drop=True)
            removed_count = original_shape[0] - df_new.shape[0]
            action_msg = f"Removed {removed_count} outlier(s) from column '{column}' using IQR method (factor={factor}). Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]"
            df = df_new

        elif operation == 'clip_outliers_iqr':
            column = params.get('column')
            factor = params.get('factor', 1.5)
            # --- Validation same as remove_outliers_iqr ---
            if not column: return jsonify({'error': 'Column required.'}), 400
            if column not in df.columns: return jsonify({'error': f'Column "{column}" not found'}), 400
            if not pd.api.types.is_numeric_dtype(df[column]): return jsonify({'error': f'Column "{column}" must be numeric.'}), 400
            try: factor = float(factor); assert factor > 0
            except: return jsonify({'error': 'Invalid factor.'}), 400
            # --- Calculation ---
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            # --- Clipping ---
            original_values = df[column].copy() # To count how many were clipped
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
            clipped_count = (original_values != df[column]).sum()
            action_msg = f"Clipped {clipped_count} outlier(s) in column '{column}' using IQR method (factor={factor}). Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]"


        # === Define Valid Output (Filtering) ===
        elif operation == 'filter_rows':
            column = params.get('column')
            condition = params.get('condition')
            value_str = params.get('value') # Value might not be needed for isnull/notnull
            filter_action = params.get('action', 'keep') # 'keep' or 'remove'

            if not column: return jsonify({'error': 'Column parameter is required.'}), 400
            if column not in df.columns: return jsonify({'error': f'Column "{column}" not found'}), 400
            if not condition: return jsonify({'error': 'Condition parameter is required.'}), 400
            if filter_action not in ['keep', 'remove']: return jsonify({'error': 'Invalid action. Must be "keep" or "remove".'}), 400

            valid_conditions = ['==', '!=', '>', '<', '>=', '<=', 'contains', 'startswith', 'endswith', 'isnull', 'notnull']
            if condition not in valid_conditions:
                 return jsonify({'error': f'Invalid condition "{condition}". Valid conditions are: {valid_conditions}'}), 400

            # Handle conditions that don't need a value
            if condition in ['isnull', 'notnull']:
                mask = df[column].isnull() if condition == 'isnull' else df[column].notnull()
                value_display = "" # No value to display
            else:
                # Conditions requiring a value
                if value_str is None: # Check if value was provided
                     return jsonify({'error': f'Value parameter is required for condition "{condition}".'}), 400

                value_display = f"'{value_str}'" # For message
                target_value = value_str # Default to string

                # Attempt type conversion based on column dtype for comparison
                col_dtype = df[column].dtype
                try:
                     if pd.api.types.is_numeric_dtype(col_dtype):
                          target_value = pd.to_numeric(value_str)
                          value_display = str(target_value)
                     elif pd.api.types.is_datetime64_any_dtype(col_dtype):
                          target_value = pd.to_datetime(value_str)
                          value_display = str(target_value)
                     elif pd.api.types.is_bool_dtype(col_dtype):
                           lowered_val = value_str.lower()
                           if lowered_val in ['true', '1', 'yes']: target_value = True
                           elif lowered_val in ['false', '0', 'no']: target_value = False
                           else: raise ValueError("Invalid boolean value")
                           value_display = str(target_value)
                     # Else: Keep as string (already assigned) for string operations
                except (ValueError, TypeError) as conv_err:
                     # If conversion fails for numeric/datetime/bool, return error as comparison likely invalid
                      return jsonify({'error': f'Could not convert value "{value_str}" to match type of column "{column}". Error: {conv_err}'}), 400

                # Build mask based on condition
                if condition == '==': mask = (df[column] == target_value)
                elif condition == '!=': mask = (df[column] != target_value)
                elif condition == '>': mask = (df[column] > target_value)
                elif condition == '<': mask = (df[column] < target_value)
                elif condition == '>=': mask = (df[column] >= target_value)
                elif condition == '<=': mask = (df[column] <= target_value)
                # String conditions (apply only if target_value is string, maybe check col type too?)
                elif condition == 'contains':
                    if not isinstance(target_value, str): return jsonify({'error': 'Contains condition requires a string value.'}), 400
                    mask = df[column].astype(str).str.contains(target_value, na=False)
                elif condition == 'startswith':
                    if not isinstance(target_value, str): return jsonify({'error': 'Startswith condition requires a string value.'}), 400
                    mask = df[column].astype(str).str.startswith(target_value, na=False)
                elif condition == 'endswith':
                    if not isinstance(target_value, str): return jsonify({'error': 'Endswith condition requires a string value.'}), 400
                    mask = df[column].astype(str).str.endswith(target_value, na=False)

            # Apply filter
            if filter_action == 'keep':
                df_new = df[mask].reset_index(drop=True)
                kept_removed_count = df_new.shape[0]
                action_word = "Kept"
            else: # remove
                df_new = df[~mask].reset_index(drop=True)
                kept_removed_count = original_shape[0] - df_new.shape[0]
                action_word = "Removed"

            action_msg = f"{action_word} {kept_removed_count} row(s) where '{column}' {condition} {value_display}."
            df = df_new


        # === Transforming and Rearranging ===
        elif operation == 'split_column':
            column = params.get('column')
            delimiter = params.get('delimiter')
            new_column_names_str = params.get('new_column_names') # Expect comma-separated string from UI

            if not column: return jsonify({'error': 'Column parameter is required.'}), 400
            if column not in df.columns: return jsonify({'error': f'Column "{column}" not found'}), 400
            if not delimiter: return jsonify({'error': 'Delimiter parameter is required.'}), 400
            if not (pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column])):
                 return jsonify({'error': 'Split column operation only applicable to text columns.'}), 400

            try:
                # Perform the split
                split_data = df[column].astype(str).str.split(delimiter, expand=True)
                num_new_cols = split_data.shape[1]

                # Determine new column names
                if new_column_names_str:
                    new_names = [name.strip() for name in new_column_names_str.split(',') if name.strip()]
                    if len(new_names) != num_new_cols:
                        return jsonify({'error': f'Provided {len(new_names)} new column names, but split resulted in {num_new_cols} columns.'}), 400
                else:
                    # Generate default names
                    new_names = [f"{column}_split_{i+1}" for i in range(num_new_cols)]

                # Check if new names conflict with existing ones (excluding original column)
                existing_cols = set(df.columns) - {column}
                conflicts = set(new_names) & existing_cols
                if conflicts:
                     return jsonify({'error': f'New column names conflict with existing columns: {list(conflicts)}'}), 400

                # Add new columns to the DataFrame
                df[new_names] = split_data
                action_msg = f"Split column '{column}' into {num_new_cols} new column(s): {', '.join(new_names)}."
                # Optional: Add parameter to drop original column `df = df.drop(columns=[column])`
            except Exception as e:
                return jsonify({'error': f'Error during split operation: {e}'}), 500

        elif operation == 'combine_columns':
            columns_to_combine = params.get('columns_to_combine') # Expect list
            new_column_name = params.get('new_column_name')
            separator = params.get('separator', '') # Default to empty string

            if not columns_to_combine or not isinstance(columns_to_combine, list) or len(columns_to_combine) < 2:
                 return jsonify({'error': 'Requires a list of at least two columns to combine.'}), 400
            if not new_column_name:
                 return jsonify({'error': 'New column name parameter is required.'}), 400
            if not all(col in df.columns for col in columns_to_combine):
                 missing = [col for col in columns_to_combine if col not in df.columns]
                 return jsonify({'error': f'Columns not found: {missing}'}), 400
            if new_column_name in df.columns:
                 # Optional: Allow overwrite? For now, prevent it.
                 return jsonify({'error': f'New column name "{new_column_name}" already exists.'}), 400

            try:
                # Convert all columns to string and join
                df[new_column_name] = df[columns_to_combine].astype(str).agg(separator.join, axis=1)
                action_msg = f"Combined columns {columns_to_combine} into '{new_column_name}' using separator '{separator}'."
                # Optional: Add parameter to drop original columns `df = df.drop(columns=columns_to_combine)`
            except Exception as e:
                return jsonify({'error': f'Error during combine operation: {e}'}), 500

        # === Other Common Operations ===

        elif operation == 'change_dtype':
            column = params.get('column')
            target_type = params.get('target_type')
            if not column: return jsonify({'error': 'Column parameter is required.'}), 400
            if column not in df.columns: return jsonify({'error': f'Column "{column}" not found'}), 400
            if not target_type: return jsonify({'error': 'Target type parameter is required.'}), 400

            original_nulls = df[column].isnull().sum()
            coerced_count = 0
            try:
                if target_type in ['int', 'integer']:
                    # Use Int64 for nullable integers
                    df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
                elif target_type in ['float', 'double', 'number']:
                    df[column] = pd.to_numeric(df[column], errors='coerce').astype(float) # Standard float
                elif target_type in ['str', 'string', 'text', 'object']:
                    df[column] = df[column].astype(str)
                elif target_type in ['datetime', 'date']:
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                elif target_type in ['bool', 'boolean']:
                     # pd.BooleanDtype() allows NA
                     df[column] = df[column].astype(pd.BooleanDtype()) # This handles 'True','False','true','false',1,0, NAs well
                elif target_type == 'category':
                    df[column] = df[column].astype('category')
                else:
                    return jsonify({'error': f'Unsupported target type: {target_type}'}), 400

                new_nulls = df[column].isnull().sum()
                coerced_count = new_nulls - original_nulls
                action_msg = f"Converted column '{column}' to type '{target_type}'."
                if coerced_count > 0:
                     action_msg += f" {coerced_count} value(s) set to null due to conversion errors."

            except Exception as e:
                # Revert if conversion fails? Less critical here as errors='coerce' handles most cases
                return jsonify({'error': f'Error converting column "{column}" to {target_type}: {e}'}), 500


        elif operation == 'rename_column':
            old_name = params.get('old_name')
            new_name = params.get('new_name')
            if not old_name: return jsonify({'error': 'Old column name parameter is required.'}), 400
            if old_name not in df.columns: return jsonify({'error': f'Column "{old_name}" not found'}), 400
            if not new_name: return jsonify({'error': 'New column name parameter is required.'}), 400
            if new_name == old_name: return jsonify({'message': 'New name is the same as the old name. No change made.', 'df_modified': False }), 200 # No change needed
            if new_name in df.columns: return jsonify({'error': f'New column name "{new_name}" already exists.'}), 400

            df.rename(columns={old_name: new_name}, inplace=True)
            action_msg = f"Renamed column '{old_name}' to '{new_name}'."


        elif operation == 'drop_columns':
            columns_to_drop = params.get('columns_to_drop') # Expect list
            if not columns_to_drop or not isinstance(columns_to_drop, list):
                 return jsonify({'error': 'Requires a list of columns to drop.'}), 400

            actual_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
            if not actual_cols_to_drop:
                 return jsonify({'error': 'None of the specified columns found in the data.'}), 400

            df = df.drop(columns=actual_cols_to_drop) # Assign back is safer than inplace
            action_msg = f"Dropped columns: {', '.join(actual_cols_to_drop)}."
            if len(actual_cols_to_drop) < len(columns_to_drop):
                 missing = [col for col in columns_to_drop if col not in actual_cols_to_drop]
                 action_msg += f" (Columns not found and ignored: {', '.join(missing)})"

        elif operation == 'replace_text':
             column = params.get('column')
             text_to_find = params.get('text_to_find')
             replace_with = params.get('replace_with', '') # Default to replace with empty string
             use_regex = params.get('use_regex', False) # Default to literal replacement

             if not column: return jsonify({'error': 'Column parameter is required.'}), 400
             if column not in df.columns: return jsonify({'error': f'Column "{column}" not found'}), 400
             if text_to_find is None: return jsonify({'error': 'Text to find parameter is required.'}), 400 # Allow empty string
             if not (pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column])):
                  return jsonify({'error': 'Replace text operation only applicable to text columns.'}), 400

             try:
                 # Ensure string type for replacement
                 df[column] = df[column].astype(str).str.replace(str(text_to_find), str(replace_with), regex=bool(use_regex))
                 mode = "regex" if use_regex else "literal"
                 action_msg = f"Replaced text '{text_to_find}' with '{replace_with}' in column '{column}' (mode: {mode})."
             except re.error as e:
                 if use_regex:
                      return jsonify({'error': f'Invalid regex pattern provided: {e}'}), 400
                 else: # Should not happen in literal mode, but just in case
                      return jsonify({'error': f'Error during text replacement: {e}'}), 500
             except Exception as e:
                  return jsonify({'error': f'Error during text replacement: {e}'}), 500

        elif operation == 'change_case':
            column = params.get('column')
            case_type = params.get('case_type') # 'lower', 'upper', 'title'
            if not column: return jsonify({'error': 'Column parameter is required.'}), 400
            if column not in df.columns: return jsonify({'error': f'Column "{column}" not found'}), 400
            if not (pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column])):
                 return jsonify({'error': 'Change case operation only applicable to text columns.'}), 400
            if case_type not in ['lower', 'upper', 'title']:
                 return jsonify({'error': 'Invalid case type. Must be "lower", "upper", or "title".'}), 400

            if case_type == 'lower': df[column] = df[column].astype(str).str.lower()
            elif case_type == 'upper': df[column] = df[column].astype(str).str.upper()
            elif case_type == 'title': df[column] = df[column].astype(str).str.title()
            action_msg = f"Converted text in column '{column}' to {case_type} case."

        elif operation == 'map_values':
            column = params.get('column')
            mapping_dict = params.get('mapping_dict') # Expect a dict/object
            if not column: return jsonify({'error': 'Column parameter is required.'}), 400
            if column not in df.columns: return jsonify({'error': f'Column "{column}" not found'}), 400
            if not mapping_dict or not isinstance(mapping_dict, dict):
                 return jsonify({'error': 'Mapping dictionary parameter is required and must be an object/dictionary.'}), 400

            try:
                 # Use replace which is generally more flexible than map for this purpose
                 df[column] = df[column].replace(mapping_dict)
                 action_msg = f"Mapped values in column '{column}' using provided dictionary."
            except Exception as e: # Handle potential type issues if dict keys/values mismatch column
                 return jsonify({'error': f'Error applying map/replace: {e}'}), 500

        elif operation == 'sort_values':
            columns_to_sort_by = params.get('columns_to_sort_by') # Expect list
            ascending = params.get('ascending', True) # Default True, can be bool or list

            if not columns_to_sort_by or not isinstance(columns_to_sort_by, list):
                 return jsonify({'error': 'Requires a list of columns to sort by.'}), 400
            if not all(col in df.columns for col in columns_to_sort_by):
                 missing = [col for col in columns_to_sort_by if col not in df.columns]
                 return jsonify({'error': f'Columns not found: {missing}'}), 400

            # Validate ascending parameter format if it's a list
            if isinstance(ascending, list) and len(ascending) != len(columns_to_sort_by):
                return jsonify({'error': 'If "ascending" is a list, its length must match the number of columns to sort by.'}), 400

            try:
                 # Use ignore_index=True to reset the index after sorting
                 df = df.sort_values(by=columns_to_sort_by, ascending=ascending, ignore_index=True)
                 asc_desc = "ascending" if ascending is True else ("descending" if ascending is False else str(ascending))
                 action_msg = f"Sorted DataFrame by columns: {', '.join(columns_to_sort_by)} ({asc_desc})."
            except Exception as e:
                 return jsonify({'error': f'Error during sorting: {e}'}), 500


        # --- Add more 'elif operation == ...' blocks here for other functions ---
        # elif operation == 'remove_outliers_iqr':
        #     # Get column, calculate Q1, Q3, IQR, filter df
        #     action_msg = "Removed outliers using IQR method..."
        # elif operation == 'rename_column':
        #     # Get old_name, new_name, perform df.rename()
        #     action_msg = "Renamed column..."

        # --- Operation not found ---
        else:
            # If operation isn't recognized, we shouldn't have added to undo.
            # This part is tricky because undo was already potentially modified.
            # Best practice: Validate operation *before* modifying undo history.
            # Quick Fix: Log it, maybe try to pop from undo? (Risky)
            app.logger.warning(f"Unknown cleaning operation received: {operation}")
            # Attempt to revert undo (may not be perfectly safe depending on add_to_undo logic)
            undo_history = session.get('undo_history', [])
            if undo_history:
                 session['undo_history'] = undo_history[:-1] # Remove the last added state

            return jsonify({'error': f'Unknown cleaning operation: {operation}'}), 400

        # 5. Store the MODIFIED DataFrame back into the session
        store_dataframe_in_session(df) # Use the helper function

        table_html, total_rows, total_columns = render_table_html(df)

        # 6. Prepare successful JSON response
        response_data = {
            'message': action_msg,
            'table_html': render_table_html(df), # Use helper to generate HTML
            'columns': df.columns.tolist(),       # Send updated column list
            'undo_redo_status': get_undo_redo_status(), # Send button states
            'total_rows': total_rows,
            'total_columns': total_columns
        }
        return jsonify(response_data), 200

    except Exception as e:
        # 7. Handle unexpected errors during the operation
        app.logger.error(f"Error during cleaning operation '{operation}': {e}", exc_info=True) # Log full traceback

        # Attempt to rollback state by restoring the last known good state from undo history
        # This prevents the session from holding a potentially corrupted intermediate state
        undo_history = session.get('undo_history', [])
        if undo_history:
             # The last item in undo_history is the state *before* the failed operation
             last_good_state_json = undo_history.pop() # Remove it as it represents the current state now
             session['current_df_json'] = last_good_state_json
             session['undo_history'] = undo_history # Save the shortened undo history
             # Don't clear redo; the failed action didn't succeed, so redo might still be valid.
        else:
             # If no undo history, we might be in trouble. Clear the current DF?
             clear_session_data() # Or handle differently

        # Send generic error back to frontend
        return jsonify({'error': f'An internal server error occurred during the operation: {str(e)}'}), 500


def suggest_missing_values(df, column, threshold=0.05): # Suggest if > 5% missing
    """Checks for missing values and suggests actions."""
    missing_count = df[column].isnull().sum()
    if missing_count > 0:
        missing_percent = missing_count / len(df)
        if missing_percent > threshold:
            return {
                'issue': 'High Missing Values',
                'suggestion': "Consider using 'Fill Missing' or 'Remove Rows with Missing'.",
                'details': f"{missing_percent:.1%} missing ({missing_count} values).",
                'action': ['fill_missing', 'remove_missing'] # Corresponding UI actions
            }
        elif missing_count > 0: # Suggest even for low counts, but maybe lower priority
             return {
                 'issue': 'Some Missing Values',
                 'suggestion': "Consider using 'Fill Missing' or 'Remove Rows with Missing'.",
                 'details': f"{missing_percent:.1%} missing ({missing_count} values).",
                 'action': ['fill_missing', 'remove_missing'],
                 'priority': 'low' # Add priority hint
             }
    return None

def suggest_whitespace(df, column, sample_size=1000, threshold=0.01): # Suggest if > 1% have whitespace issues in sample
    """Checks for leading/trailing whitespace in string columns."""
    if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
        sample_df = df[[column]].dropna().sample(n=min(sample_size, len(df.dropna())), random_state=1)
        if not sample_df.empty:
            stripped_equals_original = sample_df[column].astype(str).apply(lambda x: x == x.strip())
            whitespace_issues_count = (~stripped_equals_original).sum()
            if whitespace_issues_count > 0:
                 whitespace_percent = whitespace_issues_count / len(sample_df)
                 if whitespace_percent > threshold:
                     return {
                         'issue': 'Leading/Trailing Whitespace',
                         'suggestion': "Use 'Trim Whitespace' to clean.",
                         'details': f"~{whitespace_percent:.1%} of sample have extra spaces.",
                         'action': ['remove_spaces']
                     }
    return None

def suggest_datetime_conversion(df, column, sample_size=500, threshold=0.8): # Suggest if > 80% of sample parse as dates
    """Suggests converting object columns that look like datetimes."""
    if pd.api.types.is_object_dtype(df[column]):
         sample_df = df[[column]].dropna().sample(n=min(sample_size, len(df.dropna())), random_state=1)
         if not sample_df.empty:
            try:
                 parsed_dates = pd.to_datetime(sample_df[column], errors='coerce')
                 parse_success_rate = parsed_dates.notnull().sum() / len(sample_df)
                 if parse_success_rate > threshold:
                      return {
                          'issue': 'Potential Datetime Column (as Text)',
                          'suggestion': "Use 'Convert to Date/Time' for proper analysis.",
                          'details': f"~{parse_success_rate:.1%} of sample parse as dates.",
                          'action': ['fix_datetime']
                      }
            except Exception:
                 pass # Ignore errors during speculative parsing
    return None

def suggest_category_conversion(df, column, unique_threshold=0.1, nunique_max=50): # Suggest if < 10% unique values (and not too many)
     """Suggests converting low-cardinality string columns to Category."""
     if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
         n_unique = df[column].nunique()
         if n_unique < nunique_max and (n_unique / len(df)) < unique_threshold:
             return {
                 'issue': 'Low Cardinality Text',
                 'suggestion': "Convert to 'Category' type for efficiency.",
                 'details': f"{n_unique} unique values.",
                 'action': ['change_dtype'], # Parameter would be 'category'
                 'priority': 'low'
             }
     return None

def suggest_outliers_iqr(df, column, threshold=0.01): # Suggest if > 1% outliers
     """Suggests checking numeric columns for outliers using IQR."""
     if pd.api.types.is_numeric_dtype(df[column]) and not pd.api.types.is_bool_dtype(df[column]):
        # Drop NAs before calculating quantiles
        col_data = df[column].dropna()
        if len(col_data) < 10: return None # Not enough data for reliable IQR

        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0: return None # Avoid division by zero or infinite bounds if data is constant

        factor = 1.5
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        outliers_count = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
        if outliers_count > 0:
            outlier_percent = outliers_count / len(df) # Use original length for percentage context
            if outlier_percent > threshold:
                 return {
                     'issue': 'Potential Outliers (IQR)',
                     'suggestion': "Review values and consider 'Remove/Clip Outliers (IQR)'.",
                     'details': f"~{outlier_percent:.1%} ({outliers_count}) values fall outside 1.5*IQR.",
                     'action': ['remove_outliers_iqr', 'clip_outliers_iqr']
                 }
     return None

def suggest_id_check(df, column, unique_threshold=0.9): # Suggest if > 90% unique
    """Suggests checking uniqueness for high-cardinality non-numeric columns."""
    if not pd.api.types.is_numeric_dtype(df[column]):
        n_unique = df[column].nunique()
        if n_unique / len(df) > unique_threshold:
             return {
                 'issue': 'Potential ID Column',
                 'suggestion': "Consider 'Check Uniqueness' if this is an identifier.",
                 'details': f"{n_unique} unique values out of {len(df)}.",
                 'action': ['check_id_uniqueness'],
                 'priority': 'low'
             }
    return None

# --- New Route for Suggestions ---

@app.route('/suggest_cleaning', methods=['GET'])
def suggest_cleaning():
    """Analyzes the data and suggests cleaning steps."""
    df = get_dataframe_from_session()
    if df is None:
        return jsonify({'error': 'No data loaded.'}), 400

    suggestions = []

    # --- Table-Level Suggestions ---
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
         suggestions.append({
             'column': '(Table-Wide)',
             'issue': 'Duplicate Rows',
             'suggestion': "Use 'Remove Duplicate Rows' if these are unintentional.",
             'details': f"{duplicate_rows} identical rows found.",
             'action': ['remove_duplicates']
         })

    # --- Column-Level Suggestions ---
    for col in df.columns:
        col_suggestions = []

        # Check Missing Values
        missing_sug = suggest_missing_values(df, col)
        if missing_sug: col_suggestions.append(missing_sug)

        # Check Whitespace (only if not mostly missing)
        if not (missing_sug and missing_sug['details'].startswith("100")) : # Avoid check if all missing
            whitespace_sug = suggest_whitespace(df, col)
            if whitespace_sug: col_suggestions.append(whitespace_sug)

        # Check for Datetime Conversion (if object type)
        datetime_sug = suggest_datetime_conversion(df, col)
        if datetime_sug: col_suggestions.append(datetime_sug)

        # Check for Category Conversion (if object/string and not potential datetime)
        if not datetime_sug:
             category_sug = suggest_category_conversion(df, col)
             if category_sug: col_suggestions.append(category_sug)

        # Check for Outliers (if numeric)
        outlier_sug = suggest_outliers_iqr(df, col)
        if outlier_sug: col_suggestions.append(outlier_sug)

        # Check for Potential IDs (if high uniqueness & not numeric)
        if not outlier_sug: # Don't check ID if it looks numeric with outliers
            id_sug = suggest_id_check(df, col)
            if id_sug: col_suggestions.append(id_sug)


        # Add column name to each suggestion for that column
        for sug in col_suggestions:
            sug['column'] = col
            suggestions.append(sug)

    # Sort suggestions (e.g., by putting high priority ones first) - Simple sort for now
    suggestions.sort(key=lambda x: 0 if x.get('priority') != 'low' else 1)


    return jsonify({'suggestions': suggestions})


# --- New Route for Auto Cleaning ---

@app.route('/auto_clean', methods=['POST'])
def auto_clean_data():
    """
    Applies a predefined set of safe, non-destructive cleaning steps.
    - Trims whitespace from string columns.
    - Attempts conversion of object columns to numeric/datetime (errors='coerce').
    - Converts low-cardinality strings to Category.
    - Converts string columns to lowercase.
    """
    # 1. Get DataFrame and add to undo
    df = get_dataframe_from_session()
    if df is None:
        return jsonify({'error': 'No data loaded.'}), 400

    current_df_json = session.get('current_df_json')
    add_to_undo(current_df_json) # Save state before cleaning

    actions_performed = [] # Keep track of what was done

    try:
        df_cleaned = df.copy() # Work on a copy

        for col in df_cleaned.columns:
            original_dtype = df_cleaned[col].dtype
            col_actions = [] # Track actions for this specific column

            # --- Step 1: Trim Whitespace (if object/string) ---
            if pd.api.types.is_object_dtype(original_dtype) or pd.api.types.is_string_dtype(original_dtype):
                 # Check if trimming actually changes anything to avoid unnecessary operations/logging
                 if (df_cleaned[col].astype(str) != df_cleaned[col].astype(str).str.strip()).any():
                      df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
                      col_actions.append("Trimmed whitespace")

            # --- Step 2 & 3: Attempt Numeric / Datetime Conversion (if object) ---
            if pd.api.types.is_object_dtype(df_cleaned[col].dtype): # Check dtype again after potential stripping
                # Attempt Numeric Conversion
                try:
                    original_nulls = df_cleaned[col].isnull().sum()
                    converted_numeric = pd.to_numeric(df_cleaned[col], errors='coerce')
                    # Check if conversion actually worked for a significant portion (heuristic)
                    # and didn't just create NaNs for everything that wasn't already NaN
                    if converted_numeric.notnull().sum() > 0 and converted_numeric.isnull().sum() < (original_nulls + 0.5 * (len(df_cleaned) - original_nulls)) :
                         # Check if it's genuinely numeric (not all NaNs) and dtype changed
                         if pd.api.types.is_numeric_dtype(converted_numeric.dtype) and converted_numeric.dtype != original_dtype :
                              # Decide int or float based on whether conversion introduced decimals
                              if (converted_numeric.dropna() % 1 == 0).all():
                                   df_cleaned[col] = converted_numeric.astype(pd.Int64Dtype()) # Use nullable Int
                                   col_actions.append("Converted to Integer")
                              else:
                                   df_cleaned[col] = converted_numeric.astype(float) # Standard float
                                   col_actions.append("Converted to Float")
                except Exception:
                    pass # Ignore errors during speculative numeric conversion

                # Attempt Datetime Conversion (only if not converted to numeric)
                if pd.api.types.is_object_dtype(df_cleaned[col].dtype): # Check dtype *again*
                     try:
                         # Be careful: pd.to_datetime can be very slow on large mixed-type columns
                         # Consider sampling or more specific checks if performance is an issue
                         converted_datetime = pd.to_datetime(df_cleaned[col], errors='coerce', infer_datetime_format=True)
                         # Check if conversion actually worked and changed the type
                         if converted_datetime.notnull().sum() > 0 and pd.api.types.is_datetime64_any_dtype(converted_datetime.dtype):
                              if converted_datetime.dtype != original_dtype:
                                   df_cleaned[col] = converted_datetime
                                   col_actions.append("Converted to Datetime")
                     except (TypeError, ValueError, OverflowError):
                         pass # Ignore errors during speculative datetime conversion
                     except Exception:
                          pass # Catch other potential broad errors


            # --- Step 4: Convert Low-Cardinality Strings to Category (if still object/string) ---
            current_dtype = df_cleaned[col].dtype # Get potentially updated dtype
            if pd.api.types.is_object_dtype(current_dtype) or pd.api.types.is_string_dtype(current_dtype):
                 n_unique = df_cleaned[col].nunique()
                 if n_unique / len(df_cleaned) < 0.5 and n_unique < 1000: # Heuristic for category
                     # Check if not already category
                     if not pd.api.types.is_categorical_dtype(current_dtype):
                         df_cleaned[col] = df_cleaned[col].astype('category')
                         col_actions.append(f"Converted to Category ({n_unique} unique)")

            # --- Step 5: Change Case (if still object/string/category) ---
            current_dtype = df_cleaned[col].dtype # Check dtype again
            if pd.api.types.is_object_dtype(current_dtype) or pd.api.types.is_string_dtype(current_dtype) or pd.api.types.is_categorical_dtype(current_dtype):
                 # Check if not already all lowercase (avoid unnecessary computation)
                 # Sample check for efficiency? For now, apply directly.
                 try:
                      # Ensure string conversion before lower(), works for category too
                      if (df_cleaned[col].astype(str) != df_cleaned[col].astype(str).str.lower()).any():
                           # For category, modifying values requires care or converting back/forth
                           if pd.api.types.is_categorical_dtype(current_dtype):
                                # Option 1: Modify categories (if pandas >= 1.3.0)
                                if hasattr(df_cleaned[col].cat, 'rename_categories'):
                                     df_cleaned[col].cat.rename_categories(str.lower, inplace=True)
                                # Option 2: Convert to string, lower, back to category (less efficient)
                                else:
                                     df_cleaned[col] = df_cleaned[col].astype(str).str.lower().astype('category')
                           else: # object or string
                                df_cleaned[col] = df_cleaned[col].astype(str).str.lower()
                           col_actions.append("Converted to lowercase")
                 except Exception as e:
                      app.logger.warning(f"Could not convert column '{col}' to lowercase: {e}")


            if col_actions:
                 actions_performed.append(f"Column '{col}': {', '.join(col_actions)}")

        # 4. Store modified DataFrame and prepare response
        store_dataframe_in_session(df_cleaned)

        summary_message = "Auto Clean complete."
        if actions_performed:
             summary_message += f" Actions performed:\n - " + "\n - ".join(actions_performed)
        else:
             summary_message += " No specific cleaning actions were automatically applied based on criteria."

        table_html, total_rows, total_columns = render_table_html(df)

        response_data = {
            'message': summary_message,
            'table_html': render_table_html(df_cleaned),
            'columns': df_cleaned.columns.tolist(),
            'undo_redo_status': get_undo_redo_status(),
            'total_rows': total_rows,
            'total_columns': total_columns
        }
        return jsonify(response_data), 200

    except Exception as e:
        app.logger.error(f"Error during Auto Clean: {e}", exc_info=True)
        # Attempt rollback
        undo_history = session.get('undo_history', [])
        if undo_history:
             last_good_state_json = undo_history.pop()
             session['current_df_json'] = last_good_state_json
             session['undo_history'] = undo_history
        return jsonify({'error': f'An internal server error occurred during Auto Clean: {str(e)}'}), 500


# --- Undo/Redo Routes ---

@app.route('/undo', methods=['POST'])
def undo():
    """Reverts to the previous state."""
    current_df_json = session.get('current_df_json')
    undo_history = session.get('undo_history', [])
    redo_history = session.get('redo_history', [])

    if not undo_history:
        return jsonify({'error': 'Nothing to undo'}), 400

    # Move current state to redo
    if current_df_json:
        redo_history.insert(0, current_df_json) # Add to beginning of redo list

    # Get last state from undo
    last_state_json = undo_history.pop()

    # Update session
    session['current_df_json'] = last_state_json
    session['undo_history'] = undo_history
    session['redo_history'] = redo_history[-10:] # Limit redo history size

    # Load DF and prepare response
    df = get_dataframe_from_session()

    table_html, total_rows, total_columns = render_table_html(df)

    response_data = {
        'message': 'Undo successful.',
        'table_html': render_table_html(df),
        'columns': df.columns.tolist() if df is not None else [],
        'undo_redo_status': get_undo_redo_status(),
        'total_rows': total_rows,
        'total_columns': total_columns
    }
    return jsonify(response_data)

@app.route('/redo', methods=['POST'])
def redo():
    """Re-applies a previously undone state."""
    current_df_json = session.get('current_df_json')
    undo_history = session.get('undo_history', [])
    redo_history = session.get('redo_history', [])

    if not redo_history:
        return jsonify({'error': 'Nothing to redo'}), 400

    # Move current state to undo
    if current_df_json:
        undo_history.append(current_df_json)

    # Get first state from redo
    next_state_json = redo_history.pop(0) # Get from beginning

    # Update session
    session['current_df_json'] = next_state_json
    session['undo_history'] = undo_history[-10:] # Limit undo history size
    session['redo_history'] = redo_history

    # Load DF and prepare response
    df = get_dataframe_from_session()

    table_html, total_rows, total_columns = render_table_html(df)

    response_data = {
        'message': 'Redo successful.',
        'table_html': render_table_html(df),
        'columns': df.columns.tolist() if df is not None else [],
        'undo_redo_status': get_undo_redo_status(),
        'total_rows': total_rows,
        'total_columns': total_columns
    }
    return jsonify(response_data)


# --- Download and Save Routes ---

@app.route('/download/<filetype>')
def download_file(filetype):
    """Downloads the current DataFrame as CSV or XLSX."""
    df = get_dataframe_from_session()
    if df is None:
        flash("No data to download.", "error")
        return redirect(url_for('clean_data_interface'))

    buffer = io.BytesIO()
    filename = f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        if filetype == 'csv':
            df.to_csv(buffer, index=False, encoding='utf-8')
            mimetype = 'text/csv'
            filename += '.csv'
        elif filetype == 'xlsx':
            # Requires openpyxl
            df.to_excel(buffer, index=False, engine='openpyxl')
            mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            filename += '.xlsx'
        else:
            flash("Invalid file type for download.", "error")
            return redirect(url_for('clean_data_interface'))

        buffer.seek(0)
        return send_file(
            buffer,
            as_attachment=True,
            download_name=filename,
            mimetype=mimetype
        )
    except Exception as e:
        flash(f"Error generating download file: {e}", "error")
        app.logger.error(f"Download Error: {e}", exc_info=True)
        return redirect(url_for('clean_data_interface'))

@app.route('/save', methods=['POST'])
def save_session():
    """Saves the current DataFrame state to a file on the server."""
    df = get_dataframe_from_session()
    if df is None:
        return jsonify({'error': 'No data to save'}), 400

    # Use existing filename or generate a new one
    save_name = request.json.get('filename')
    if save_name:
         # Basic sanitization, consider more robust validation
         filename = secure_filename(save_name).replace('..', '') # Prevent path traversal
         if not filename.endswith('.pkl'):
             filename += '.pkl'
    else:
         # Generate unique name if not provided
         filename = f"session_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d')}.pkl"

    filepath = os.path.join(app.config['SAVED_SESSIONS_FOLDER'], filename)

    try:
        # Use pickle to preserve DataFrame structure and types
        df.to_pickle(filepath)
        session['saved_filename'] = filename # Store filename in session to show user
        flash(f"Session saved as '{filename}'. You can reopen it later.", "success")
        return jsonify({
            'message': f"Session saved as {filename}",
            'saved_filename': filename
        })
    except Exception as e:
        flash(f"Error saving session: {e}", "error")
        app.logger.error(f"Save Error: {e}", exc_info=True)
        return jsonify({'error': f'Error saving session: {str(e)}'}), 500


@app.route('/open_saved/<filename>')
def open_saved_session(filename):
    """Loads a previously saved DataFrame session."""
    # IMPORTANT: Security check - ensure filename is safe
    safe_filename = secure_filename(filename)
    if not safe_filename.endswith('.pkl'): # Enforce extension
         flash("Invalid file format.", "error")
         return redirect(url_for('index'))

    filepath = os.path.join(app.config['SAVED_SESSIONS_FOLDER'], safe_filename)

    if not os.path.exists(filepath):
        flash(f"Saved session '{safe_filename}' not found.", "error")
        return redirect(url_for('index'))

    try:
        df = pd.read_pickle(filepath)
        clear_session_data() # Clear any existing session data
        store_dataframe_in_session(df)
        session['source_info'] = f"Saved Session: {safe_filename}"
        session['saved_filename'] = safe_filename # Remember which file is loaded
        flash(f"Loaded saved session '{safe_filename}'.", "info")
        return redirect(url_for('clean_data_interface'))
    except Exception as e:
        flash(f"Error loading saved session: {e}", "error")
        app.logger.error(f"Open Saved Error: {e}", exc_info=True)
        # Consider removing the corrupted file?
        return redirect(url_for('index'))

# --- Run Application ---
if __name__ == '__main__':
    # Use threaded=False if you encounter issues with session consistency
    # during rapid AJAX requests in development. In production use a proper WSGI server.
    app.run(debug=True, threaded=True)
