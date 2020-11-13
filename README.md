# xRcovid

This repository contains the supplementary material for the article "Using Artificial Intelligence to Improve COVID-19 Rapid Diagnostic Test Result Interpretation" by D.-A. Mendels et al, PNAS 2020. This is all the material that needs to be used to train a CNN on Covid-19 serological RDTs (rapid diagnostic tests). There are two types of resources:
- Design files
- Images.

The design files are for 3D printing a simple cradle (or support) for iPhone. Though not necessary per se, it is what makes capturing large numbers of RDTs in little time possible, with constant and consistent illumination and distance from the RDT. The sub-folder has a general view for assembly, and the STL files that are needed to 3D print the shapes. For reference, we printed all those on a Creality Ender 3 in a few hours.

The Images folder has two subfolders, which in turn have each two subfolders. We split the case of two-well RDTs and single RDTs. Each have positive and negative subfolders, containing images of positive and negative RDTs. The RDTs have been read by two analysts, and that reading has been confirmed on the image. importantly, the control bar of the RDT is always located at the center of the image, approximately. For training a CNN, we recommend that the SINGLE set is used, with a split 80-10-10 for Training-Validation-Set respectively. It is easy to split the two-well RDTs into two halves (left and right), and run the trained CNN on them for validation: this is the way we managed that problem.

Due to the large size of the folder, the data is available at the following repository: https://mega.nz/folder/9H4DTC5K#J3kU-Dg42svTev3R9v-3ig

Or: you can browse the Images_RDTs folder, each zip file can be downloaded separately. You are also encouraged to save those files to your own Mega cloud repository (free storage up to 50GB).

In case something goes wrong, please contact support@xrapid-group.com
