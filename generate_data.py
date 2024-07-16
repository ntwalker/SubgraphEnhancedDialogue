import pandas as pd
import random
import json
import re
import time
from datetime import datetime
from faker import Faker

class FakeOrg():
    
    def __init__(self, group_names, rooms, meeting_chunks, lang="en_UK", n_people=30, n_events=60):
    
        self.fake = Faker([lang])
        self.group_names = group_names
        self.rooms = rooms
        self.meeting_chunks = meeting_chunks
        
        self.people = []
        self.people_calendars = dict()
        self.events = []


        self.groups = {g: [] for g in group_names}
        self.group_calendars = {g: [] for g in group_names}
        self.room_calendars = {r: [] for r in rooms}

        for i in range(n_people):
            self.create_person()

        for i in range(n_events):
            self.create_event()
        #self.people_calendars = {p["name"]: [] for p in people}

    def create_person(self):
        
        name = "-"
        while len(name.split()) != 2 or "-" in name:
            name = self.fake.name() # Ensure only first and last name (no e.g. "Dr John Doe, Jane Doe MD")
        email = name.lower().replace(" ", ".") + "@company.email"
        phone = "+49" + " 00" + self.fake.msisdn()[:8]
        group_assignment = random.choice(self.group_names)
        office = str(random.randrange(1,9)) + str(random.randrange(0,9)) + str(random.randrange(0,9))
        profile = {"name": name, "index": len(self.people), "email": email, "phone": phone, "group": group_assignment, "office": office, "calendar": []}
           
        self.people.append(profile)
        self.people_calendars[name] = []   
        self.groups[group_assignment].append(profile)

    def create_event(self, organizer=None, n_attendees=2):

        if organizer == None:
            organizer = random.choice(self.people)
            
        full_date = self.fake.date_time_between(start_date='now', end_date='+7d') #Generate fake date
        event_date = full_date.date() # Select only the date

        event_hour = random.choice([i for i in range(8, 16)]) # Make sure the event start hour is between 8 and 4
        event_minutes = random.choice([0, 30]) # Make some events start and end on a half hour

        event_start = full_date.time().replace(hour=event_hour, minute=event_minutes) # Start time, rounded to the hour
        event_end = event_start.replace(hour=event_hour+1, minute=event_minutes).strftime("%H:%M") # End time, rounded to the hour
        event_start = event_start.strftime("%H:%M")

        event_id = self.fake.sha256()[:10]
        event_group = organizer["group"] # Assign event to the same group as the organizer
        event_location = random.choice(self.rooms) # Choose a room randomly for the meeting
        
        #event_name = random.choice([fake.bs().split()[2], event_group]) + random.choice(self.meeting_chunks) #Fake event name
        event_name = self.fake.bs().split()[2] + random.choice(self.meeting_chunks) #Fake event name

        #event_group = random.choice(self.group_names) # Choose a group randomly for the meeting

        group_members = self.groups[event_group]
        
        event_attendees = [organizer["name"]]
        attendee_profiles = [a for a in random.sample(group_members, k=min(n_attendees, len(self.groups[event_group]))) if not a["name"] == organizer["name"]]
        attendee_indices = {a["name"]: a["index"] for a in attendee_profiles} # To find in people list
        attendee_indices[organizer["name"]] = organizer["index"] # Add index of organizer
        event_attendees.extend([a["name"] for a in attendee_profiles])

        event_profile = {"id": event_id, "name": event_name, "date": str(event_date), "start_time": str(event_start), "end_time": str(event_end), 
                         "location": event_location, "organizer": organizer["name"], "attendees": event_attendees, 
                         "group": event_group}

        for a in event_attendees:
            self.people_calendars[a].append(event_profile)
            self.people[attendee_indices[a]]["calendar"].append(event_profile)
        self.group_calendars[event_group].append(event_profile)
        self.room_calendars[event_location].append(event_profile)
        
        self.events.append(event_profile)
        
    def save_data(self, prefix=""):
        
        with open("people_calendars.json", "w") as file:
            file.write(json.dumps(self.people_calendars))
            
        with open("group_calendars.json", "w") as file:
            file.write(json.dumps(self.group_calendars))
            
        with open("room_calendars.json", "w") as file:
            file.write(json.dumps(self.room_calendars))
            
        
        print("Data Saved")
        
    def save_events(self, prefix=""):
        
        with open("event_calendars.json", "w") as file:
            file.write(json.dumps(self.events))     
        
        print("Events Saved")

def create_calendar():
    n_people = random.randint(30,60)
    n_events = random.randint(30,60)

    print("Generated %d people, %d events" %(n_people, n_events))

    group_names = ["Administration", "Biology", "Climate", "Engineering", "Healthcare", "Mathematics", "Philosophy", "Statistics"]
    rooms = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Sigma", "Omega"]
    meeting_chunks = [" team meeting", " status update", " discussion", " project meeting", " seminar", " workshop"]

    org_data = FakeOrg(group_names, rooms, meeting_chunks, lang="en_US", n_people=n_people, n_events=n_events)
    org_data.save_data() #Create summary of Org Data

    events_table = pd.DataFrame(org_data.events, columns=['id,' 'attendees', 'start_time', 'end_time', 'isodate', 'date'])
    events_table['attendees'] = events_table['attendees'].apply(lambda x: ", ".join(x))
    events_table['start_time'] = events_table['start_time'].apply(lambda x: re.sub(r"\:00$", "", x))
    events_table['end_time'] = events_table['end_time'].apply(lambda x: re.sub(r"\:00$", "", x))
    events_table['isodate'] = events_table['date']
    events_table['date'] = events_table['date'].apply(lambda x: date.fromisoformat(x).strftime('%d-%m-%y'))
    events_table.drop(["id"], axis=1, inplace=True)

    people_table = pd.DataFrame(org_data.people)
    people_table.drop(['calendar'], axis=1, inplace=True)
    people_table.drop(['id'], axis=1, inplace=True)
    people_table.sort_values(by=['name'], inplace=True)
    
    room_data = []
    for key, value in org_data.rooms.items():
        for item in value['calendar']:
            room_data.append({**{'location': key}, **value})
    room_table = pd.DataFrame(room_data)
    #room_table['attendees'] = room_table['attendees'].apply(lambda x: ", ".join(x))
    room_table.drop(['id'], axis=1, inplace=True)

    return events_table, people_table, room_table, org_data

# Task Generator
def generate_task(task_templates, org):

    fake = Faker()

    task = random.sample(task_templates, 1)[0]
    filled_task = task

    ents = re.findall(r"_.+?_", task)

    visitor = random.choice((org.people))
    calendar = visitor["calendar"]
    if len(calendar) == 0:
        org.create_event(organizer=visitor)
    others = [person["name"] for person in org.people if not person["name"] == visitor]
    
    for i in range(len(ents)):
        if ents[i] == "_NewPerson_":
            filled_task = re.sub(r"_NewPerson_", visitor["name"], filled_task, 1)
        if ents[i] == "_NewEvent_":
            meeting_chunks = [" team meeting", " status update", " discussion", " project meeting", " seminar", " workshop"]
            event_name = fake.bs().split()[2] + random.choice(meeting_chunks) #Fake event name
            filled_task = re.sub(r"_NewEvent_", event_name, filled_task, 1)
        if ents[i] == "_Person_":
            pers = random.choice(others)
            filled_task = re.sub(r"_Person_", pers, filled_task, 1)
        if ents[i] == "_Event_":
            event_name = random.choice(org.events)["name"]
            filled_task = re.sub(r"_Event_", "\"" + event_name + "\"", filled_task , 1)
        if ents[i] == "_Room_":
            room = random.sample(sorted(org.location), 1)[0]
            filled_task = re.sub(r"_Room_", room, filled_task, 1)
        if ents[i] == "_Group_":
            group = random.choice([g for g in org.groups.keys()])
            #group = random.sample(sorted(org.group), 1)[0]
            filled_task = re.sub(r"_Group_", group, filled_task, 1)
        if ents[i] == "_Time_":
            c_time = str(random.randint(8, 16)) + ":00"
            filled_task = re.sub(r"_Time_", c_time, filled_task, 1)
        if ents[i] == "_CalendarEvent_":
            calendar = visitor["calendar"]
            event_name = random.choice(calendar)["name"]
            filled_task = re.sub(r"_CalendarEvent_", event_name, filled_task, 1)

    return filled_task


tasks = ["You’re a visitor at a new office named _NewPerson_. You’re supposed to meet with someone named _Person_. Find out if they’re currently available and set up a meeting with them as soon as they are available.",
    "You’re a new employee named _NewPerson_. You’re supposed to meet with someone named _Person_. Find out if they are available at _Time_ tomorrow and set up a meeting with them at or as near to this time as possible.",
    "You’re a researcher called _NewPerson_ and have been invited to a meeting called _CalendarEvent_. Find out when and where this meeting will take place.",
    "You’re visiting a research institute and want to find _Person_ today. As _NewPerson_, find out if they are in a meeting, and if they are available set up a time to meet with them.",
    "You’ve been invited as _NewPerson_ to talk to members of a local research group in _Group_. Find out what events this group has tomorrow and who the organizers are. Set up a meeting with each of them that day", 
    "You’re attending a meeting called _CalendarEvent_, but you’ve realized you need to make sure _Person_ attends. As _NewPerson_, add them to the meeting if they aren’t already in it.",
    "You’re attending a meeting called _CalendarEvent_. As _NewPerson_, find out when the event starts, how long it lasts, where it will be, and who is organizing it.",
    "You’re attending a meeting called _CalendarEvent_. As _NewPerson_, find out who is organizing the event and what other events are on their calendar.",
    "You’re attending a meeting called _CalendarEvent_. As _NewPerson_, find out where this meeting will be and what other events will be in that location that day.",
    "Your name is _NewPerson_, you want to meet with a member of the _Group_ group. Find someone from this group who is currently available and set up a meeting with them as soon as possible in an available meeting room.",
    "Your name is _NewPerson_, you want to meet with a member of the _Group_ group. Find someone from this group who is currently available and set up a meeting with them as soon as possible in an available meeting room.",
    "Your name is _NewPerson_, you’re supposed to attend a meeting called _CalendarEvent_. Find out when it is and where it will be held. Then, find out if you have any meetings after that meeting.",
    "You are _NewPerson_ and have been invited to a meeting called _CalendarEvent_. Find out when and where this meeting will take place.",
    "You are _NewPerson_, and you want to arrange an hour long meeting called _NewEvent_. Make an hour long meeting in an available conference room, and invite _Person_ and _Person_.",
    "Your name is _NewPerson_, and you’d like to talk to _Person_. Find out where their office is and what floor of the building it’s on.",
    "You are _NewPerson_. Make a meeting with _Person_ in an available meeting room at _Time_ tomorrow, or at their nearest availability.",
    "You are _NewPerson_. Make a meeting with _Person_ in their office at _Time_ tomorrow, or at their nearest availability.",
    "Your name is _NewPerson_, you want to get in contact with a member of the _Group_ group. Find a member of this group and get their phone number and email address.",
    "Your name is _NewPerson_, you want to get in contact with the organizer of _Event_. Find out their phone number, email address and what group they work in.",
    "You are _NewPerson_. You’re looking for the contact information of _Person_. Find out what group they work in and get their email address.",
    "You are _NewPerson_. Make a meeting at _Time_ tomorrow with two members of the _Group_ group, or the nearest available time.",

    "You are _NewPerson_, and you want to arrange a meeting called _NewEvent_. Make a half hour long meeting in an available conference room, and invite _Person_ and _Person_.",
    "You are _NewPerson_, and you want to arrange a meeting called _NewEvent_. Make a half hour long meeting in an available conference room, and invite _Person_ and at least one member of the _Group_ group.",
    "You are _NewPerson_ and need to find the time and date of a meeting called _Event_. Find out when what day and time it is and what group is holding it.",
    "You are _NewPerson_. Find out when the meeting called _CalendarEvent_ is, and make a half hour long meeting with the same attendees immediately after it.",
    "You are _NewPerson_. Find out who is organizing _Event_, and find out the room number for their office and what group they work in.",
    "You are _NewPerson_. For tomorrow, arrange a meeting with a member of the _Group_ group and a member of the _Group_ group.",
    "Your name is _NewPerson_, and you need to join a meeting called _Event_. Ask the robot to add you to the meeting, then make a new meeting afterwards with the organizer of that event and one other attendee.",
    "Your name is _NewPerson_, and you need to make a meeting called _NewEvent_ after your meeting _CalendarEvent_. Create the new meeting the same attendees of that event, excluding the organizer.",
    "As _NewPerson_, find out what events you are organizing, if any. If you are organizing any events today or tomorrow, cancel them. Then, set up a meeting called _NewEvent_ with one member from the _Group_ group.",
    "As _NewPerson_, find out what events you are attending, if any. Set up a meeting called _NewEvent_ with the organizer of one of these events at _Time_, or with anyone else if you have no events.",
    "Your name is _NewPerson_, and you need to make sure that _Person_ attends a meeting of yours. Find out what events are on your calendar, and add them to one of them. If there are no events, create a meeting with the other person called _NewEvent_.",
    "You are _NewPerson_, and you need to postpone your meetings from today to tomorrow. Find out what events you have today, and move them to tomorrow at the same time slots, or the nearest possible time slots.",
    "As _NewPerson_, cancel any morning meetings on your calendar today, and cancel any afternoon meetings you have tomorrow.",
    "As _NewPerson_, find a meeting on your calendar and move it to one day later at the same time, then add _Person_ if they are available at that time.",
    "You are a new employee named _Person_. Find out what your new office is, then set up an introductory meeting with _Person_ in your office tomorrow morning.",
    "You are _NewPerson_, and you want to invite _Person_ and two members of the _Group_ group to a meeting. Arrange a meeting called _Event_ tomorrow morning with these three people.",
    "You are _NewPerson_, and you need to set up a two hour long meeting called _NewEvent_ tomorrow morning. Invite three available members from the _Group_ and _Group_ groups. If you have any conflicting events on your calendar, cancel them.",


]

if __name__ == "__main__":

    data = []

    group_names = ["Administration", "Biology", "Climate", "Engineering", "Healthcare", "Mathematics", "Philosophy", "Statistics"]
    rooms = ["Meeting Room Alpha", "Meeting Room Beta", "Meeting Room Gamma", "Meeting Room Delta", "Meeting Room Epsilon", "Meeting Room Zeta", "Meeting Room Sigma", "Meeting Room Omega"]
    meeting_chunks = [" team meeting", " status update", " discussion", " project meeting", " seminar", " workshop", " follow up", " conference"]
    
    for i in range(1000):
        org = FakeOrg(group_names, rooms, meeting_chunks)
        task = generate_task(tasks, org)
        data.append({"task" : task, "data": {"events": org.events, "people": org.people, "rooms": org.rooms, "groups": org.groups}})

    with open("data/tasks_testdata.json", "w") as file:
        json.dump(data, file, indent=4)
