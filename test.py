from datetime import datetime

def age_from_birthdate(birthdate_str, current_date_str='2023-09-07'):
    birth_date = datetime.strptime(birthdate_str, '%Y-%m-%d')
    current_date = datetime.strptime(current_date_str, '%Y-%m-%d')

    age = current_date.year - birth_date.year - (
                (current_date.month, current_date.day) < (birth_date.month, birth_date.day))

    return age


birthdate = '2001-08-30'
print(age_from_birthdate(birthdate))