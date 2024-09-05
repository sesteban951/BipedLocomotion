import numpy as np
import csv
from pydrake.all import LeafSystem, TriggerType, PublishEvent

class ContactLogger(LeafSystem):
    
    def __init__(self, plant, update_interval, file_name):
        LeafSystem.__init__(self)
        
        # Store the plant to know the bodies
        self.plant = plant

        # Declare input port for ContactResults
        self.DeclareAbstractInputPort(
            "contact_results", plant.get_contact_results_output_port().Allocate())

        # Declare a periodic event to log ContactResults every update_interval seconds
        self.DeclarePeriodicEvent(
            period_sec=update_interval, 
            offset_sec=0.0, 
            event=PublishEvent(
                trigger_type=TriggerType.kPeriodic, 
                callback=self.DoCalcDiscreteVariableUpdates))
        
        # Initialize CSV file and writer
        self.file_name = file_name
        with open(self.file_name, mode='w', newline='') as file:
            self.writer = csv.writer(file)
            self.writer.writerow(["Time", "BodyA", "BodyB", "ContactPoint", "NormalForce"])

    def LogContactResults(self, contact_results, current_time):

        # cap the floating poitn for time
        current_time = round(current_time, 5)

        # get the number of contact pairs
        num_contact_pairs = contact_results.num_point_pair_contacts()
        
        # Process contacts
        if num_contact_pairs > 0:
            for i in range(num_contact_pairs):

                # extract contact info
                contact_info = contact_results.point_pair_contact_info(i)
                contact_point = contact_info.contact_point()
                normal_force = contact_info.contact_force()
                bodyA_id = contact_info.bodyA_index()
                bodyB_id = contact_info.bodyB_index()
                bodyA_name = self.plant.get_body(bodyA_id).name()
                bodyB_name = self.plant.get_body(bodyB_id).name()

                # log the entry        
                log_entry = [current_time, bodyA_name, bodyB_name, contact_point, normal_force]
                self._append_to_csv(log_entry)

        # No contacts
        else:
            log_entry = [current_time, "None", "None", np.array([0,0,0]), np.array([0,0,0])]
            self._append_to_csv(log_entry)
        
    def _append_to_csv(self, log_entry):
        with open(self.file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(log_entry)

    def DoCalcDiscreteVariableUpdates(self, context, event=None):
        contact_results = self.get_input_port(0).Eval(context)
        current_time = context.get_time()
        self.LogContactResults(contact_results, current_time)