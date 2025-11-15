def baseInfo(file_path):
    # Open the EDF file in binary mode and read the first 256 bytes (EDF header)
    with open(file_path, 'rb') as f:
        header_bytes = f.read(256)  # Only read the first 256 bytes

    # Check if the file is at least 256 bytes long
    if len(header_bytes) < 256:
        raise ValueError("File is less than 256 bytes; not a complete EDF header")

    # Parse basic fields from the EDF header
    edf_version = header_bytes[0:8].decode('ascii').strip()       # EDF version
    patient_id = header_bytes[8:88].decode('ascii').strip()       # Patient ID field
    recording_id = header_bytes[88:168].decode('ascii').strip()   # Recording ID field
    startdate = header_bytes[168:176].decode('ascii').strip()     # Start date (dd.mm.yy)
    starttime = header_bytes[176:184].decode('ascii').strip()     # Start time (hh.mm.ss)
    header_bytes_length = header_bytes[184:192].decode('ascii').strip()  # Header length
    reserved = header_bytes[192:236].decode('ascii').strip()      # Reserved bytes
    num_data_records = header_bytes[236:244].decode('ascii').strip()      # Number of data records
    duration_of_record = header_bytes[244:252].decode('ascii').strip()    # Duration of a record in seconds
    num_signals = header_bytes[252:260].decode('ascii').strip()           # Number of signals

    # Parse subfields from Patient ID
    patient_parts = patient_id.split()
    patient_number = patient_parts[0] if len(patient_parts) > 0 else None
    sex = patient_parts[1] if len(patient_parts) > 1 and patient_parts[1] in ['M', 'F'] else None
    # Extract age if present in the format "Age:XX"
    age_str = next((p for p in patient_parts if p.startswith("Age:")), None)
    age = int(age_str[4:]) if age_str else None

    # Build a lightweight JSON dictionary with essential info
    minimal_info = {
        "patient_id": patient_number,
        "sex": sex,
        "age": age,
        "start_date": startdate,
        "start_time": starttime,
        "data_duration": f"[0 - {int(num_data_records) * float(duration_of_record)}]",  # Total duration in seconds
    }

    # Remove fields that are None
    minimal_info = {k: v for k, v in minimal_info.items() if v is not None}
    return minimal_info


def get_age_factor(age, age_factors):
    """
    Return the age factor for a given age based on predefined ranges.
    age_factors: list of dicts, each with keys "min_age" and "max_age"
    """
    for factor in age_factors:
        if factor["min_age"] <= age < factor["max_age"]:
            return factor
    return None  # Return None if no range matches
